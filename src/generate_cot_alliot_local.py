import argparse
import json
import pickle
import random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

dataset_paths = {
    'aime': '../data/aime_1983_2023/train.jsonl',
    'gsm8k': '../data/gsm8k/train.jsonl',
    'math': '../data/math/train.jsonl'
}

def load_model(model_name: str):
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B: {"<think>": 151648, "</think>": 151649}
    # Qwen/Qwen3-8B: {"<think>": 151667, "</think>": 151668}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
#        dtype="auto",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def extract_solution_gsm8k(text: str) -> str:
    """
    Extracts the final solution value from a string where the solution is marked after '####'.
    """
    marker = "\n####"
    return text.split(marker, 1)[1].strip()


def generate_cot(prompt, tokenizer, model, max_new_tokens=32768, top_k=20):
    """
    Returns:
        - Prompt+CoT tokens
        - Prompt+CoT token-ids
        - CoT token-level *top-k* probs        (list[list[float]], k each step)
        - CoT token-level *top-k* log-probs    (list[list[float]], k each step)
        - CoT token-level entropies            (float per step; exact)
        - CoT token-level </think> probabilities (float per step; exact)
        - prompt length
    """
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True
    ).to(model.device)

    # Try to resolve the </think> token id dynamically, fall back to Qwen3 id.
    try:
        eot_id = tokenizer.convert_tokens_to_ids("</think>")
        if eot_id is None or eot_id < 0:
            eot_id = 151668
    except Exception:
        eot_id = 151668  # Qwen/Qwen3-8B
    # NOTE: DeepSeek-R1-Distill-Qwen-1.5B uses 151649 for </think>.

    output = model.generate(
        **inputs,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_logits=True,
        stop_strings='</think>',
        # NOTE: the below parameters are recommended for Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0
    )

    input_and_output_ids = output.sequences
    input_len = inputs['input_ids'].shape[1]

    # We'll keep only top-k for probs/log-probs
    output_topk_probs: list[list[float]] = []
    output_topk_log_probs: list[list[float]] = []
    output_entropies: list[float] = []
    output_eot_probs: list[float] = []

    for logits in output.logits:  # logits shape: [1, vocab]
        # logsumexp denominator for exact probabilities
        denom = torch.logsumexp(logits, dim=-1)  # [1]

        # Top-k by logit (works better than top-k on probs)
        topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)  # [1, k]
        # Exact top-k log-probs / probs w.r.t. full distribution
        topk_log_probs = (topk_vals - denom.unsqueeze(-1)).squeeze(0)  # [k]
        topk_probs = torch.exp(topk_log_probs)                          # [k]

        # Store as plain Python lists on CPU (small, pickle-friendly)
        output_topk_log_probs.append(topk_log_probs.detach().cpu().tolist())
        output_topk_probs.append(topk_probs.detach().cpu().tolist())

        # Exact entropy (computed from full distribution, not stored)
        log_probs_full = F.log_softmax(logits, dim=-1)
        probs_full = log_probs_full.exp()
        entropy = -(probs_full * log_probs_full).sum(dim=-1)  # [1]
        output_entropies.append(float(entropy.item()))

        # Exact prob of </think>
        eot_log_prob = (logits[0, eot_id] - denom[0])
        output_eot_probs.append(float(torch.exp(eot_log_prob).item()))

    # Flatten token ids
    input_and_output_token_ids = input_and_output_ids[0].tolist()

    # Find end of </think> if present and slice everything to that point
    try:
        end_of_think_id = len(input_and_output_token_ids) - input_and_output_token_ids[::-1].index(eot_id)
        input_and_cot_ids = input_and_output_token_ids[:end_of_think_id]
        cot_probs       = output_topk_probs[:end_of_think_id - input_len]
        cot_log_probs   = output_topk_log_probs[:end_of_think_id - input_len]
        cot_entropies   = output_entropies[:end_of_think_id - input_len]
        cot_eot_probs   = output_eot_probs[:end_of_think_id - input_len]
    except ValueError:
        # No </think> found; keep all
        input_and_cot_ids = input_and_output_token_ids
        cot_probs       = output_topk_probs
        cot_log_probs   = output_topk_log_probs
        cot_entropies   = output_entropies
        cot_eot_probs   = output_eot_probs

    # Detokenize each token individually
    input_and_cot_tokens = [tokenizer.decode([tid], skip_special_tokens=False)
                            for tid in input_and_cot_ids]

    return (input_and_cot_tokens, input_and_cot_ids,
            cot_probs, cot_log_probs, cot_entropies, cot_eot_probs, input_len)

#def generate_cot(prompt, tokenizer, model, max_new_tokens=32768):
#    """
#    Returns:
#        - Prompt+CoT tokens
#        - Prompt+CoT token-ids
#        - CoT token-level entropies
#        - CoT token-level </think> probabilities
#        - CoT token-level </think> log-probabilities
#        - prompt length
#    """
#    messages = [
#        {"role": "user", "content": prompt}
#    ]
#    inputs = tokenizer.apply_chat_template(
#        messages,
#        add_generation_prompt=True,
#        tokenize=True,
#        return_dict=True,
#        return_tensors="pt",
#        enable_thinking=True
#    ).to(model.device)
#
#    # Generate output with logits
#    output = model.generate(
#        **inputs,
#        tokenizer=tokenizer,
#        max_new_tokens=max_new_tokens,
#        return_dict_in_generate=True,
#        output_logits=True,
#        stop_strings='</think>',
#        # NOTE: the below parameters are recommended for Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B
#        temperature=0.6,
#        top_p=0.95,
#        top_k=20,
#        min_p=0
#    )
#
#    # Get the prompt length
#    input_and_output_ids = output.sequences
#    input_len = inputs['input_ids'].shape[1]
#
#    # Compute the entropy and </think> probability at for each token (assumes batch size = 1)
#    output_probs     = []
#    output_log_probs = []
#    output_entropies = []
#    output_eot_probs = []
#    for logits in output.logits:
#        probs = F.softmax(logits, dim=-1)           # [batch_size, vocab_size]
#        output_probs.append(probs.detach().cpu())
#        log_probs = F.log_softmax(logits, dim=-1)   # [batch_size, vocab_size]
#        output_log_probs.append(log_probs.detach().cpu())
#        entropy = -(probs * log_probs).sum(dim=-1)  # [batch_size]
#        output_entropies.append(entropy.item())
#
#        # </think> token's (log)probability
#        output_eot_prob = probs[:, 151668]          # [batch_size]
#        output_eot_probs.append(output_eot_prob.item())
#
#    # Flatten tensors
#    input_and_output_token_ids = input_and_output_ids[0].tolist()
#
#    # parsing thinking content
#    try:
#        # rindex finding 151668 (</think>) --  corresponds to position index of </think> + 1
#        end_of_think_id = len(input_and_output_token_ids) - input_and_output_token_ids[::-1].index(151668)
#        input_and_cot_ids = input_and_output_token_ids[:end_of_think_id]
#        cot_probs     = output_probs[:end_of_think_id-input_len]
#        cot_log_probs = output_log_probs[:end_of_think_id-input_len]
#        cot_entropies = output_entropies[:end_of_think_id-input_len]
#        cot_eot_probs = output_eot_probs[:end_of_think_id-input_len]
#    except ValueError:
#        input_and_cot_ids = input_and_output_token_ids
#        cot_probs     = output_probs
#        cot_log_probs = output_log_probs
#        cot_entropies = output_entropies
#        cot_eot_probs = output_eot_probs
#
#    # Detokenize each token individually
#    input_and_cot_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_and_cot_ids]
#
#    return input_and_cot_tokens, input_and_cot_ids, cot_probs, cot_log_probs, cot_entropies, cot_eot_probs, input_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B', choices=['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen3-8B'])
    parser.add_argument('--dataset', type=str, required=True, choices=['aime', 'gsm8k', 'math'])
    parser.add_argument('--subject', type=str, required=False, choices=['Algebra', 'Prealgebra'], help='Required if `dataset==math`')
    parser.add_argument('--level', type=int, required=False, choices=[1, 2, 3, 4, 5], help='Required if `dataset==math`')
    parser.add_argument('--output_dir', type=str, default='./data/outputs/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    assert args.dataset in dataset_paths, f"Dataset {args.dataset} not found in dataset_paths."
    assert not (args.dataset == 'math' and (args.subject is None or args.level is None)), "Subject and level must be specified for the math dataset."

    set_seed(args.seed)

    if args.dataset == 'math':
        save_dir = Path(args.output_dir) / f"{args.model_name.split('/')[1]}/{args.dataset}/{args.subject}/level_{args.level}"
    else:
        save_dir = Path(args.output_dir) / f"{args.model_name.split('/')[1]}/{args.dataset}"

    save_dir.mkdir(parents=True, exist_ok=True)

    # load the dataset
    with open(dataset_paths[args.dataset], 'r') as f:
        data = [json.loads(line) for line in f]

    prompts = []
    for item in data:
        if args.dataset == 'gsm8k':
            id = item['idx']
            question = item['question']
            answer = extract_solution_gsm8k(item['answer'])
        elif args.dataset == 'aime':
            id = item['id']
            question = item['question']
            answer = item['answer']
        elif args.dataset == 'math':
            if item['subject'] != args.subject or item['level'] != f"Level {args.level}":
                continue
            id = item['unique_id']
            question = item['problem']
            answer = item['answer']
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        prompts.append(
            {
                'id': id,
                'question': question,
                'answer': answer
            }
        )

    # select a subset of the dataset for analysis
    if len(prompts) > 200:
        prompts = random.sample(prompts, 200)
    model, tokenizer = load_model(model_name=args.model_name)


    # COT generation
    outputs = []
    for item in tqdm(prompts, total=len(prompts), desc="Generating COTs"):
        prompt = item['question']
        # adhere to the suggested use
        prompt += ' Please reason step by step, and put your final answer within \\boxed{}'

        input_and_cot_tokens, input_and_cot_ids, cot_probs, cot_log_probs, cot_entropies, cot_eot_probs, input_len = generate_cot(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=32768
        )

        print(len(input_and_cot_tokens), len(input_and_cot_ids), len(cot_probs), len(cot_log_probs), len(cot_entropies), len(cot_eot_probs), input_len)

        outputs.append(
            {
                'id': item['id'],
                'answer': item['answer'],
                'input_and_cot_tokens': input_and_cot_tokens,
                'input_and_cot_ids': input_and_cot_ids,
                'cot_probs': cot_probs,
                'cot_log_probs': cot_log_probs,
                'cot_entropies': cot_entropies,
                'cot_eot_probs': cot_eot_probs,
                'input_len': input_len
            }
        )

    # Save `outputs`
    with open(save_dir / f"cots.pkl", "wb") as f:
        pickle.dump(outputs, f)
