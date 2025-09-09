# train.py
import argparse
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from model import PromptCompressionWrapper

# --- optional wandb import (safe) ---
try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class PromptQADataset(Dataset):
    """
    Expects a HuggingFace dataset with columns: 'context' (prompt), 'instruction' (question), 'response' (answer).
    The dataset should follow the Dolly-15k format.
    """
    def __init__(self, path: str, max_items: int = None):
        # Load the Dolly dataset from Hugging Face
        dataset = load_dataset(path)
        # Use the "train" split by default (adjust for validation/test splits as needed)
        self.items = dataset["train"]

        if max_items is not None:
            self.items = self.items[:max_items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.items[idx]
        # Map the columns as needed
        return {
            "prompt": item["context"],       # context -> prompt
            "question": item["instruction"], # instruction -> question
            "answer": item["response"],      # response -> answer
        }


def set_all_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    prompts = [b["prompt"] for b in batch]
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]
    return {"prompt": prompts, "question": questions, "answer": answers}


def train_one_epoch(
    model: PromptCompressionWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    threshold_inference: float,
    lambda_length: float,
    entropy_beta: float,
    baseline_decay: float,
    grad_accum_steps: int = 1,
    max_batches: int = None,
    log_every: int = 50,
    # --- wandb options ---
    base_step: int = 0,
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    wandb_log_every: Optional[int] = None,
) -> Tuple[Dict[str, float], int]:
    """
    REINFORCE objective with:
      reward = -NLL(answer|ctx)  - lambda_length * kept_ratio
      loss   = - (reward - baseline) * mean(log pi(a_t)) - entropy_beta * H
    Only compressor parameters are updated.

    Returns:
      (epoch_stats_dict, steps_taken)
    """
    device = model.device
    model.train()
    # freeze decoder (safety in case)
    for p in model.decoder.parameters():
        p.requires_grad_(False)

    ema_baseline = 0.0
    step = 0
    wb_every = wandb_log_every if wandb_log_every is not None else log_every

    meters = dict(loss=0.0, reward=0.0, nll=0.0, kept_ratio=0.0, entropy=0.0, count=0)

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        rewards = []
        policy_losses = []
        entropies_total = []
        kept_ratios = []
        nll_vals = []

        # Process each sample independently (compression decisions are variable-length)
        for j in range(len(batch["prompt"])):
            prompt = batch["prompt"][j]
            question = batch["question"][j]
            answer = batch["answer"][j]

            # Sample compression decisions
            comp_out = model.compress_prompt(prompt, question, sample=True, threshold=threshold_inference)
            kept_text = comp_out.kept_text
            kept_ratio = comp_out.kept_ratio

            # Task reward via frozen decoder NLL (lower CE => higher reward)
            nll, ans_len = model.nll_of_answer(kept_text, question, answer)
            # NOTE: nll is already the mean CE over answer tokens (decoder labels masking handles this).
            nll = nll.detach()
            reward_task = -nll

            # Length penalty encourages short prompts
            reward = reward_task - lambda_length * kept_ratio

            # Sum log-probs and entropies; normalize per decision for scale stability
            logp_sum = 0.0
            H_sum = 0.0
            num_decisions = 0
            for lp, ent in zip(comp_out.log_probs, comp_out.entropies):
                logp_sum = logp_sum + lp.sum()
                H_sum = H_sum + ent.sum()
                num_decisions += lp.numel()

            if num_decisions > 0:
                logp_sum = logp_sum / num_decisions  # normalize PG term per decision
                H_sum = H_sum / num_decisions        # entropy per decision
            entropies_total.append(H_sum)

            # REINFORCE policy gradient
            advantage = reward - ema_baseline
            policy_loss = -(advantage * logp_sum)

            rewards.append(reward)
            nll_vals.append(nll)
            kept_ratios.append(torch.tensor(kept_ratio, device=device))
            policy_losses.append(policy_loss)

        # batch aggregation
        policy_loss = torch.stack(policy_losses).mean()
        entropy_term = torch.stack(entropies_total).mean()
        # encourage exploration
        loss = policy_loss - entropy_beta * entropy_term

        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Update moving baseline
        with torch.no_grad():
            batch_reward_mean = torch.stack(rewards).mean().item()
            ema_baseline = baseline_decay * ema_baseline + (1 - baseline_decay) * batch_reward_mean

        # Logging (stdout)
        meters["loss"] += loss.item()
        meters["reward"] += torch.stack(rewards).mean().item()
        meters["nll"] += torch.stack(nll_vals).mean().item()
        meters["kept_ratio"] += torch.stack(kept_ratios).mean().item()
        meters["entropy"] += entropy_term.item()
        meters["count"] += 1

        # Logging (wandb)
        if wandb_run is not None:
            wb_step = base_step + step + 1
            wandb_run.log({
                "train/loss": loss.item(),
                "train/reward": torch.stack(rewards).mean().item(),
                "train/nll": torch.stack(nll_vals).mean().item(),
                "train/kept_ratio": torch.stack(kept_ratios).mean().item(),
                "train/entropy": entropy_term.item(),
                "train/baseline": ema_baseline,
            }, step=wb_step)

        if (step + 1) % log_every == 0:
            avg = {k: meters[k] / max(1, meters["count"]) for k in ["loss", "reward", "nll", "kept_ratio", "entropy"]}
            print(f"[step {step+1}] "
                  f"loss={avg['loss']:.4f} reward={avg['reward']:.4f} "
                  f"nll={avg['nll']:.4f} kept={avg['kept_ratio']:.3f} H={avg['entropy']:.2f} "
                  f"baseline={ema_baseline:.4f}")

        step += 1

    if meters["count"] == 0:
        return {}, step

    stats = {k: meters[k] / meters["count"] for k in ["loss", "reward", "nll", "kept_ratio", "entropy"]}
    return stats, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, required=True, help="Dataset path on HuggingFace Datasets (e.g., 'databricks/databricks-dolly-15k')")
    ap.add_argument("--decoder", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--compressor", type=str, default="xlm-roberta-xl")
    ap.add_argument("--save_dir", type=str, default="ckpt_compressor")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)  # small; the head is simple but encoder is big
    ap.add_argument("--lambda_length", type=float, default=0.5, help="penalize kept_ratio")
    ap.add_argument("--entropy_beta", type=float, default=1e-3, help="encourage exploration")
    ap.add_argument("--baseline_decay", type=float, default=0.95)
    ap.add_argument("--threshold_inference", type=float, default=0.5)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_batches_per_epoch", type=int, default=None)

    # --- wandb flags ---
    ap.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="prompt-compressor")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)
    ap.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags, e.g. 'qwen,roberta,rl'")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_log_every", type=int, default=None, help="Defaults to --log_every inside the loop")

    args = ap.parse_args()

    set_all_seed(args.seed)

    # Initialize wandb (optional)
    wb_run = None
    if args.use_wandb:
        if wandb is None:
            print("[wandb] WARNING: --use_wandb set but 'wandb' is not installed. Continuing without logging.")
        else:
            if args.wandb_mode == "disabled":
                os.environ["WANDB_DISABLED"] = "true"
            else:
                os.environ.pop("WANDB_DISABLED", None)
            tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
            wb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=tags,
                mode=args.wandb_mode,
                config=vars(args),
            )

    # Load the dataset
    dataset = PromptQADataset(args.train_path, max_items=args.max_items)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    model = PromptCompressionWrapper(
        decoder_name_or_path=args.decoder,
        compressor_name_or_path=args.compressor,
    )

    # Train **only** the compressor params
    trainable_params = [p for p in model.compressor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    os.makedirs(args.save_dir, exist_ok=True)

    # Log parameter counts once
    if wb_run is not None:
        total_comp_params = sum(p.numel() for p in model.compressor.parameters())
        trainable_comp_params = sum(p.numel() for p in model.compressor.parameters() if p.requires_grad)
        wb_run.log({
            "params/compressor_total": total_comp_params,
            "params/compressor_trainable": trainable_comp_params,
        }, step=0)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        stats, steps_taken = train_one_epoch(
            model, loader, optimizer,
            threshold_inference=args.threshold_inference,
            lambda_length=args.lambda_length,
            entropy_beta=args.entropy_beta,
            baseline_decay=args.baseline_decay,
            grad_accum_steps=args.grad_accum,
            max_batches=args.max_batches_per_epoch,
            log_every=50,
            base_step=global_step,
            wandb_run=wb_run,
            wandb_log_every=args.wandb_log_every,
        )
        global_step += steps_taken

        if stats:
            print(f"Epoch {epoch} stats:", {k: round(v, 4) for k, v in stats.items()})
            if wb_run is not None:
                wb_run.log({f"epoch/{k}": v for k, v in stats.items()}, step=global_step)
                wb_run.log({"epoch/number": epoch}, step=global_step)

        # Save compressor (just the compressor module for deployment flexibility)
        save_path = os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        model.compressor.encoder.save_pretrained(save_path)  # encoder
        model.compressor.tokenizer.save_pretrained(save_path)
        # Also save the head weights
        torch.save(model.compressor.state_dict(), os.path.join(save_path, "compressor_state.pt"))
        print(f"Saved compressor to {save_path}")

        # Optionally log checkpoint path (no artifact upload by default)
        if wb_run is not None:
            wb_run.log({"checkpoint/path": save_path}, step=global_step)

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()

