# model.py
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForCausalLM,
)


@dataclass
class CompressionOutput:
    # Per-chunk bookkeeping for training/inference
    kept_text: str
    keep_probs: List[torch.Tensor]           # list of [seq_len_chunk] probabilities
    actions: Optional[List[torch.Tensor]]    # list of sampled 0/1 actions (training)
    log_probs: Optional[List[torch.Tensor]]  # log pi(a) per token (training)
    entropies: Optional[List[torch.Tensor]]  # Bernoulli entropy per token (training)
    kept_ratio: float                        # total_kept / total_prompt_tokens (not counting question)
    meta: Dict                               # assorted metadata (counts, splits, etc.)


class PromptCompressor(nn.Module):
    """
    A minimal 'RoBERTa-XL-style' encoder + 1D tokenwise head that outputs keep logits per token.
    Default backbone is 'xlm-roberta-xl' (widely available). You can swap to any AutoModel encoder.
    """
    def __init__(
        self,
        compressor_name_or_path: str = "xlm-roberta-xl",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(compressor_name_or_path)
        self.encoder = AutoModel.from_pretrained(compressor_name_or_path)
        hidden_size = self.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # keep logit per token
        )

        # expose tokenizer for chunk-aware compression
        self.tokenizer = AutoTokenizer.from_pretrained(compressor_name_or_path, use_fast=True)
        if self.tokenizer.sep_token is None:
            # ensure a separator exists (Roberta uses </s>)
            if self.tokenizer.eos_token:
                self.tokenizer.sep_token = self.tokenizer.eos_token
            else:
                # Last resort: add a custom sep token
                self.tokenizer.add_special_tokens({"sep_token": "<SEP>"})
                self.encoder.resize_token_embeddings(len(self.tokenizer))

    @property
    def max_length(self) -> int:
        # Conservative cap for safety
        ml = getattr(self.tokenizer, "model_max_length", 512)
        if ml is None or ml > 1000000000000:
            ml = 512
        return int(ml)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns keep logits of shape [B, T, 1]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, T, H]
        keep_logits = self.head(hidden)     # [B, T, 1]
        return keep_logits

    def tokenwise_keep(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample: bool,
        threshold: float = 0.5,
        force_keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute per-token keep probabilities and either sample actions (training) or threshold (inference).
        - input_ids, attention_mask: [1, T]
        - force_keep_mask: [1, T] booleans indicating tokens that must be kept (e.g., specials, question)
        Returns:
        - keep_probs: [T]
        - keep_mask: [T] booleans after sampling/threshold + force-keep applied
        - log_prob: [T] of chosen action (only when sample=True), else None
        - entropy: [T] Bernoulli entropy (only when sample=True), else None
        """
        with torch.set_grad_enabled(self.training):
            logits = self.forward(input_ids, attention_mask)  # [1, T, 1]
        logits = logits.squeeze(-1).squeeze(0)                # [T]
        keep_probs = torch.sigmoid(logits)                    # [T]

        if sample:
            dist = torch.distributions.Bernoulli(probs=keep_probs.clamp(1e-6, 1-1e-6))
            actions = dist.sample()                           # [T] in {0,1}
            log_probs = dist.log_prob(actions)               # [T]
            entropies = -(keep_probs * (keep_probs.clamp_min(1e-8)).log() +
                          (1 - keep_probs) * ((1 - keep_probs).clamp_min(1e-8)).log())
            keep_mask = actions.bool()
        else:
            actions = log_probs = entropies = None
            keep_mask = (keep_probs >= threshold)

        if force_keep_mask is not None:
            keep_mask = keep_mask | force_keep_mask.squeeze(0).bool()

        return keep_probs.detach(), keep_mask, log_probs, entropies


class PromptCompressionWrapper(nn.Module):
    """
    Ties the compressor and a frozen decoder LLM together.
    Provides:
      - compress_prompt(...): to turn (prompt, question) -> compressed_prompt_text
      - reward model utilities for training via NLL on the frozen decoder
    """
    def __init__(
        self,
        decoder_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        compressor_name_or_path: str = "xlm-roberta-xl",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Trainable compressor
        self.compressor = PromptCompressor(compressor_name_or_path).to(self.device)

        # 2) Frozen decoder
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name_or_path, dtype=torch.bfloat16).to(self.device)
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder.eval()

        # Tokenizers
        self.comp_tok = self.compressor.tokenizer
        self.dec_tok = AutoTokenizer.from_pretrained(decoder_name_or_path, use_fast=True)
        if self.dec_tok.pad_token is None:
            # For CE/NLL we want a pad_token id
            self.dec_tok.pad_token = self.dec_tok.eos_token

        # Heuristics
        self.min_keep_tokens = 1

    def _split_into_chunks(self, ids: List[int], max_len: int) -> List[List[int]]:
        """
        Split token ids (no attention mask needed) into chunks no longer than max_len.
        Tries to preserve BOS/EOS on chunk edges if present.
        """
        if len(ids) <= max_len:
            return [ids]
        chunks = []
        start = 0
        while start < len(ids):
            end = min(start + max_len, len(ids))
            chunks.append(ids[start:end])
            start = end
        return chunks

    def _force_keep_mask_for_specials_and_question(
        self,
        comp_ids: torch.Tensor,
        prompt_len: int,
        question_len: int,
        special_ids: set
    ) -> torch.Tensor:
        """
        Builds a [1, T] boolean mask forcing keep for:
          - all special tokens
          - the entire question segment (the last 'question_len' tokens)
        """
        bsz, T = comp_ids.shape
        mask = torch.zeros((bsz, T), dtype=torch.bool, device=comp_ids.device)
        # specials
        for sid in special_ids:
            mask |= (comp_ids == sid)
        # question segment
        if question_len > 0:
            mask[:, -question_len:] = True
        return mask

    def compress_prompt(
        self,
        prompt_text: str,
        question_text: str,
        sample: bool = False,
        threshold: float = 0.5,
        temperature: float = 1.0,  # for possible future use; not used directly here
    ) -> CompressionOutput:
        """
        Compresses PROMPT ONLY while allowing compressor to read [prompt || SEP || question].
        Returns decoded 'kept_text' (for prompt only) and training stats if sample=True.
        """
        device = self.device
        # Tokenize prompt and question separately to know boundaries
        ptoks = self.comp_tok(prompt_text, add_special_tokens=True)
        qtoks = self.comp_tok(question_text, add_special_tokens=True)

        prompt_ids = ptoks["input_ids"]
        question_ids = qtoks["input_ids"]

        # Compose joint input: [prompt_ids (w/ specials)] + [sep] + [question_ids (w/ specials)]
        sep_id = self.comp_tok.sep_token_id or self.comp_tok.eos_token_id
        if sep_id is None:
            # create a SEP token if truly absent
            sep_id = self.comp_tok.convert_tokens_to_ids("<SEP>")

        joint_ids = prompt_ids + [sep_id] + question_ids
        # We'll only allow dropping among indices that belong to 'prompt_ids' (excluding its specials if you want).
        # For simplicity, we allow the head to score ALL tokens but we force-keep the question span.

        # Chunk by compressor max length
        max_len = self.compressor.max_length
        chunks = self._split_into_chunks(joint_ids, max_len=max_len)

        kept_text_parts: List[str] = []
        keep_probs_list, actions_list, log_probs_list, entropies_list = [], [], [], []

        total_prompt_tokens = 0
        total_kept_prompt_tokens = 0

        # Identify special ids for force-keep: cls/eos/bos/pad/sep if present
        special_ids = set([
            tid for tid in [
                self.comp_tok.cls_token_id,
                self.comp_tok.bos_token_id,
                self.comp_tok.eos_token_id,
                self.comp_tok.pad_token_id,
                self.comp_tok.sep_token_id,
            ] if tid is not None
        ])

        # Build offsets to know in which chunk the question span lies.
        # Simpler approach: rebuild prompt length + sep + question length per chunk.
        running = 0
        prompt_total_len = len(prompt_ids)
        sep_len = 1
        question_total_len = len(question_ids)

        for chunk_ids in chunks:
            T = len(chunk_ids)
            # Determine how much of this chunk belongs to (prompt | sep | question)
            # Compute span intersections
            chunk_start = running
            chunk_end = running + T

            # global spans:
            prompt_span = (0, prompt_total_len)                  # [0, prompt_total_len)
            sep_span = (prompt_total_len, prompt_total_len + 1)  # [p, p+1)
            question_span = (prompt_total_len + 1, prompt_total_len + 1 + question_total_len)

            def span_intersection(a, b):
                s = max(a[0], b[0]); e = min(a[1], b[1])
                return (max(0, e - s), s, e) if e > s else (0, s, s)

            prompt_in_chunk_count, ps, pe = span_intersection((chunk_start, chunk_end), prompt_span)
            sep_in_chunk_count, ss, se = span_intersection((chunk_start, chunk_end), sep_span)
            q_in_chunk_count, qs, qe = span_intersection((chunk_start, chunk_end), question_span)

            # local indices (relative to chunk)
            prompt_local_start = ps - chunk_start
            question_local_start = qs - chunk_start

            # Prepare tensors
            input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            # Force keep mask: specials + question span in chunk
            force_keep = self._force_keep_mask_for_specials_and_question(
                input_ids, prompt_in_chunk_count, q_in_chunk_count, special_ids
            )

            # Forward through compressor head
            keep_probs, keep_mask, log_probs, entropies = self.compressor.tokenwise_keep(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sample=sample,
                threshold=threshold,
                force_keep_mask=force_keep,
            )

            # Enforce at least one prompt token kept within this chunk (fallback)
            # Identify prompt-local region in chunk
            if prompt_in_chunk_count > 0:
                prompt_region = torch.zeros(T, dtype=torch.bool, device=device)
                prompt_region[prompt_local_start: prompt_local_start + prompt_in_chunk_count] = True
                # Are any prompt tokens kept?
                if not (keep_mask & prompt_region).any():
                    # Keep the top-prob prompt token
                    pr_probs = keep_probs.clone()
                    pr_probs[~prompt_region] = -1.0  # mask out non-prompt
                    top_idx = int(torch.argmax(pr_probs).item())
                    keep_mask[top_idx] = True

                total_prompt_tokens += int(prompt_in_chunk_count)
                total_kept_prompt_tokens += int((keep_mask & prompt_region).sum().item())

            # Decode only kept tokens, but **exclude question** tokens from the returned prompt text
            # Keep-only ids:
            kept_ids = [tid for tid, keep in zip(chunk_ids, keep_mask.tolist()) if keep]
            # Now remove any ids that are part of the question span (local indices)
            if q_in_chunk_count > 0:
                q_local_mask = torch.zeros(T, dtype=torch.bool, device=device)
                q_local_mask[question_local_start: question_local_start + q_in_chunk_count] = True
                kept_ids = [tid for i, tid in enumerate(chunk_ids) if keep_mask[i].item() and not q_local_mask[i].item()]

            # Decode kept prompt-only tokens:
            if len(kept_ids) > 0:
                kept_text_parts.append(self.comp_tok.decode(kept_ids, skip_special_tokens=True))

            keep_probs_list.append(keep_probs)
            if sample:
                actions_list.append((keep_mask.to(keep_probs.dtype)))
                log_probs_list.append(log_probs)
                entropies_list.append(entropies)

            running += T

        kept_text = " ".join([s for s in kept_text_parts if s.strip() != ""]).strip()
        kept_ratio = (total_kept_prompt_tokens / max(1, total_prompt_tokens)) if total_prompt_tokens > 0 else 1.0

        return CompressionOutput(
            kept_text=kept_text,
            keep_probs=keep_probs_list,
            actions=actions_list if sample else None,
            log_probs=log_probs_list if sample else None,
            entropies=entropies_list if sample else None,
            kept_ratio=kept_ratio,
            meta=dict(
                total_prompt_tokens=total_prompt_tokens,
                total_kept_prompt_tokens=total_kept_prompt_tokens,
            )
        )

    # --------- Decoder-side utilities ---------

    def _format_for_decoder(self, compressed_prompt: str, question: str) -> str:
        # You can change this template as needed.
        # Keep it consistent across training and inference.
        if compressed_prompt:
            return f"{compressed_prompt}\n\nQuestion: {question}\nAnswer:"
        else:
            return f"Question: {question}\nAnswer:"

    @torch.no_grad()
    def generate(
        self,
        prompt_text: str,
        question_text: str,
        threshold: float = 0.5,
        **gen_kwargs
    ) -> str:
        self.eval()
        comp_out = self.compress_prompt(prompt_text, question_text, sample=False, threshold=threshold)
        dec_input_text = self._format_for_decoder(comp_out.kept_text, question_text)
        enc = self.dec_tok(dec_input_text, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            **gen_kwargs
        )
        outputs = self.decoder.generate(**enc, **gen_kwargs)
        full_text = self.dec_tok.decode(outputs[0], skip_special_tokens=True)
        # Return only the part after "Answer:" if present
        if "Answer:" in full_text:
            return full_text.split("Answer:", 1)[-1].strip()
        return full_text

    def nll_of_answer(
        self,
        compressed_prompt: str,
        question_text: str,
        answer_text: str,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute negative log-likelihood (cross-entropy) of the ground-truth answer
        under the frozen decoder, conditioned on (compressed_prompt, question).
        Returns (mean_ce, answer_ntokens).
        """
        dec_input_text = self._format_for_decoder(compressed_prompt, question_text)
        # Context tokens
        ctx = self.dec_tok(dec_input_text, return_tensors="pt", add_special_tokens=True).to(self.device)
        # Answer tokens (labels)
        ans = self.dec_tok(answer_text, return_tensors="pt", add_special_tokens=False).to(self.device)

        # Construct a single sequence: [ctx] + [answer]; labels only on the answer part
        input_ids = torch.cat([ctx["input_ids"], ans["input_ids"]], dim=1)  # [1, Tctx + Tans]
        attention_mask = torch.cat([ctx["attention_mask"], ans["attention_mask"]], dim=1)
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[:, ctx["input_ids"].shape[1]:] = input_ids[:, ctx["input_ids"].shape[1]:]  # teacher forcing on answer

        with torch.no_grad():
            out = self.decoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss  # mean CE over labeled tokens

        ans_len = int(ans["input_ids"].shape[1])
        return loss, ans_len

