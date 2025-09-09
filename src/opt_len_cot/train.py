# train.py
import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from model import PromptCompressionWrapper


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
) -> Dict[str, float]:
    """
    REINFORCE objective with:
      reward = -NLL(answer|ctx)  - lambda_length * kept_ratio
      loss   = - (reward - baseline) * sum(log pi(a_t)) - entropy_beta * H
    Only compressor parameters are updated.
    """
    device = model.device
    model.train()
    # freeze decoder (safety in case)
    for p in model.decoder.parameters():
        p.requires_grad_(False)

    ema_baseline = 0.0
    step = 0

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
            # normalize by answer length to keep reward scale consistent
            nll = nll.detach()
            reward_task = -nll

            # Length penalty encourages short prompts
            reward = reward_task - lambda_length * kept_ratio

            # Sum of log probs over all chunk tokens (only where actions were sampled)
            # Also collect entropy for regularization
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

        # Logging
        meters["loss"] += loss.item()
        meters["reward"] += torch.stack(rewards).mean().item()
        meters["nll"] += torch.stack(nll_vals).mean().item()
        meters["kept_ratio"] += torch.stack(kept_ratios).mean().item()
        meters["entropy"] += entropy_term.item()
        meters["count"] += 1

        if (step + 1) % log_every == 0:
            avg = {k: meters[k] / max(1, meters["count"]) for k in ["loss", "reward", "nll", "kept_ratio", "entropy"]}
            print(f"[step {step+1}] "
                  f"loss={avg['loss']:.4f} reward={avg['reward']:.4f} "
                  f"nll={avg['nll']:.4f} kept={avg['kept_ratio']:.3f} H={avg['entropy']:.2f} "
                  f"baseline={ema_baseline:.4f}")

        step += 1

    if meters["count"] == 0:
        return {}
    return {k: meters[k] / meters["count"] for k in ["loss", "reward", "nll", "kept_ratio", "entropy"]}


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
    args = ap.parse_args()

    set_all_seed(args.seed)

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

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        stats = train_one_epoch(
            model, loader, optimizer,
            threshold_inference=args.threshold_inference,
            lambda_length=args.lambda_length,
            entropy_beta=args.entropy_beta,
            baseline_decay=args.baseline_decay,
            grad_accum_steps=args.grad_accum,
            max_batches=args.max_batches_per_epoch,
        )
        if stats:
            print(f"Epoch {epoch} stats:", {k: round(v, 4) for k, v in stats.items()})

        # Save compressor (just the compressor module for deployment flexibility)
        save_path = os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        model.compressor.encoder.save_pretrained(save_path)  # encoder
        model.compressor.tokenizer.save_pretrained(save_path)
        # Also save the head weights
        torch.save(model.compressor.state_dict(), os.path.join(save_path, "compressor_state.pt"))
        print(f"Saved compressor to {save_path}")


if __name__ == "__main__":
    main()

