import random
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def load_model(model_name: str):
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B: {"<think>": 151648, "</think>": 151649}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def find_sublist_indices(long_list, short_list):
    indices = []
    len_short = len(short_list)
    for i in range(len(long_list) - len_short + 1):
        if long_list[i:i+len_short] == short_list:
            indices.append(i)
    return indices

def chunk_cot(input_and_cot_tokens, input_and_cot_ids, cot_eot_probs, input_len, method):
    """
    Splits a CoT into chunks with 4 different methods:
        - "keywords": split at specific keywords ("wait", "alternatively").
        - "newlines": split at double-newline ("\n\n").
        - "uniform": split into (64) ~equal-sized chunks.
        - "eot_threshold": split at punctuation marks with </think> probability above a certain threshold.
    """

    assert method in ['keywords', 'eot_threshold', 'uniform', 'newlines']
    
    if method == 'keywords':
        keywords = ['wait', 'alternatively']
        indices = [i for i, token in enumerate(input_and_cot_tokens) if token.lower().strip() in keywords and i > input_len]
    
    elif method == 'newlines':
        delimiter = '\n\n'
        indices = [i for i, token in enumerate(input_and_cot_tokens) if delimiter in token and i > input_len]
    
    elif method == 'uniform':
        cot_len = len(input_and_cot_tokens) - input_len
        if cot_len < 128:
            num_chunks=4
        elif cot_len < 512:
            num_chunks=8
        elif cot_len < 1024:
            num_chunks=16
        elif cot_len < 2048:
            num_chunks=32
        else:
            num_chunks=64
        chunk_size = max(1, cot_len // num_chunks)
        indices = [input_len + (i + 1) * chunk_size for i in range(num_chunks - 1)]
    
    elif method == 'eot_threshold':
        punctuations=['.', '?', '!', '.\n', '?\n', '!\n', '.\n\n', '?\n\n', '!\n\n']
        punct_indices = []
        i = input_len
        while i < len(input_and_cot_tokens):
            if input_and_cot_tokens[i] in punctuations:
                j = i
                while j + 1 < len(input_and_cot_tokens) and input_and_cot_tokens[j+1] in punctuations:
                    j += 1
                if j < len(input_and_cot_tokens) - 1:
                    # take last punctuation in the run
                    # that is not the last token
                    punct_indices.append(j)
                i = j + 1
            else:
                i += 1

        if punct_indices:
            punct_indices = np.array(punct_indices)
            punct_eot_log_probs = np.log(np.array(cot_eot_probs)[punct_indices - input_len + 1])

            # threshold = mean+std of eot_log_prob of punctuations
            log_threshold = punct_eot_log_probs.mean() + punct_eot_log_probs.std()

            indices = punct_indices[np.where(punct_eot_log_probs > log_threshold)[0]].tolist()

        else:
            # when no punctuation found
            indices = []
        
    else:
        raise ValueError(f"Unknown method: {method}")

    # Add sentinel end indices
    if input_and_cot_tokens[-1] == '</think>':
        ends = [input_len + 1] + indices + [len(input_and_cot_ids)-1]
    else:
        ends = [input_len + 1] + indices + [len(input_and_cot_ids)]

    # Slice and filter out empty lists
    chunk_end_ids = []
    chunks_cumulative = []
    for end in ends:
        if end > 0 and input_and_cot_ids[:end]:
            chunk_end_ids.append(end)
            chunks_cumulative.append(input_and_cot_ids[:end])

    return chunk_end_ids, chunks_cumulative

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COT for a dataset")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B', choices=['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen3-8B'])
    parser.add_argument('--dataset', type=str, required=True, choices=['aime', 'gsm8k', 'math', 'hotpotqa'])
    parser.add_argument('--subject', type=str, required=False, choices=['Algebra', 'Prealgebra'], help='Required if `dataset==math`')
    parser.add_argument('--level', type=int, required=False, choices=[1, 2, 3, 4, 5], help='Required if `dataset==math`')
    parser.add_argument('--output_dir', type=str, default='/scratch/saydalie/llm_reasoning/data/outputs/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    assert not (args.dataset == 'math' and (args.subject is None or args.level is None)), "Subject and level must be specified for the math dataset."

    set_seed(args.seed)

    if args.dataset == 'math':
        save_dir = Path(args.output_dir) / f"{args.model_name.split('/')[1]}/{args.dataset}/{args.subject}/level_{args.level}"
    else:
        save_dir = Path(args.output_dir) / f"{args.model_name.split('/')[1]}/{args.dataset}"

    # load the dataset
    with open(save_dir / "cots.pkl", "rb") as f:
        cot_data = pickle.load(f)

    if len(cot_data) > 200:
        cot_data = random.sample(cot_data, 200)

    model, tokenizer = load_model(model_name=args.model_name)

    # COT generation
    if (save_dir / "chunks_answer_probs.pkl").exists():
        with open(save_dir / "chunks_answer_probs.pkl", "rb") as f:
            data_output = pickle.load(f)
    else:
        data_output = []

    already_processed_ids = [d['id'] for d in data_output]

    for item in tqdm(cot_data, total=len(cot_data), desc="Generating Answer Probs with Chunks"):
        if item['id'] in already_processed_ids:
            continue

        answer_ids = tokenizer([item['answer']]).input_ids[0]
        answer_indices = find_sublist_indices(item['input_and_cot_ids'], answer_ids)

        target_outputs = {
            'with_eot': f"\n</think>\n\n\\boxed{{{item['answer']}}}",
            'with_eot_conditioned': f"\n</think>\n\n\\boxed{{{item['answer']}}}", # [6:]
            'without_eot': f"\\boxed{{{item['answer']}}}",
            'without_eot_conditioned': f"\\boxed{{{item['answer']}}}", # [3:]
            'without_eot_nor_boxed': item['answer']
        }

        chunk_methods = {}
        # for method in ['keywords', 'eot_threshold', 'uniform', 'newlines']:
        for method in ['keywords', 'eot_threshold']:
            chunk_end_ids, chunks_cumulative = chunk_cot(
                input_and_cot_tokens=item['input_and_cot_tokens'],
                input_and_cot_ids=item['input_and_cot_ids'],
                cot_eot_probs=item['cot_eot_probs'],
                input_len=item['input_len'],
                method=method
            )

            chunk_outputs = []
            for chunk_end_id, chunk in zip(chunk_end_ids, chunks_cumulative):

                # Get the probability of <answer>
                answer_log_probs = {}
                for target_output_method, target_output in target_outputs.items():
                    target_output_ids = tokenizer.encode(target_output, add_special_tokens=False)
                    inputs = torch.tensor([chunk + target_output_ids], device=model.device)

                    with torch.no_grad():
                        outputs = model(inputs)
                        logits = outputs.logits  # shape: [1, seq_len, vocab_size]

                    # Extract logits for positions where target tokens should be predicted
                    # These positions start right after the end of input_ids
                    logits_for_targets = logits[:, len(chunk)-1:, :][:, :len(target_output_ids), :]

                    # Joint log probability
                    log_probs = torch.log_softmax(logits_for_targets, dim=-1)
                    token_log_probs = log_probs[0, torch.arange(len(target_output_ids)), target_output_ids]
                    if target_output_method == 'with_eot_conditioned':
                        joint_log_prob = token_log_probs[6:].sum().item()  
                    elif target_output_method == 'without_eot_conditioned':
                        joint_log_prob = token_log_probs[3:].sum().item()
                    else:
                        joint_log_prob = token_log_probs.sum().item()
                    answer_log_probs[target_output_method] = joint_log_prob

                chunk_outputs.append({
                    'chunk_end_id': chunk_end_id,
                    'answer_log_probs': answer_log_probs
                })

            chunk_methods[method] = chunk_outputs

        data_output.append(
            {
                'id': item['id'],
                'answer': item['answer'],
                'answer_indices': answer_indices,
                'chunks': chunk_methods
            }
        )

        # Save `data_output` periodically
        with open(save_dir / f"chunks_answer_probs.pkl", "wb") as f:
            pickle.dump(data_output, f)