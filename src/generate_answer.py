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

def extract_boxed_solution(text: str) -> str | None:
    """
    Return the content of the last LaTeX-style \boxed{...} in `text`.
    Handles nested braces (e.g. \boxed{\frac{1}{2}}) and returns None if
    no complete \boxed{...} is found.
    """
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None

    start = idx + len(marker)
    depth = 1
    i = start
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1

    # Reached end without closing brace
    return None

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
    truncation_prompt = "\n\nTime is up.\n\nGiven the time I've spent and the approaches I've tried, I should stop thinking and formulate a final answer based on what I already have.\n</think>\n\n"

    # COT generation
    if (save_dir / "chunks_answer.pkl").exists():
        with open(save_dir / "chunks_answer.pkl", "rb") as f:
            data_output = pickle.load(f)
    else:
        data_output = []

    already_processed_ids = [d['id'] for d in data_output]

    for item in tqdm(cot_data, total=len(cot_data), desc="Generating Answer with Chunks"):
        if item['id'] in already_processed_ids:
            continue

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

            batch_size = 2  # adjust depending on GPU memory
            num_return_sequences = 3
            chunk_outputs = []

            for i in tqdm(range(0, len(chunk_end_ids), batch_size)):
                batch_chunk_end_ids = chunk_end_ids[i:i+batch_size]
                batch_chunks = chunks_cumulative[i:i+batch_size]

                batch_inputs = []
                for chunk in batch_chunks:
                    inp = chunk + tokenizer.encode(truncation_prompt, add_special_tokens=False)
                    batch_inputs.append(inp)

                batch_inputs = tokenizer.pad(
                    {"input_ids": batch_inputs},
                    return_tensors="pt",
                    padding_side='left'
                ).to(model.device)

                # Generate `num_return_sequences` answers per input in the batch
                with torch.no_grad():
                    outputs = model.generate(
                        **batch_inputs,
                        max_new_tokens=4096,
                        num_return_sequences=num_return_sequences,
                    )

                # Split outputs: since num_return_sequences > 1, HuggingFace arranges them in order
                # Example: [x1..xn, y1..yn, z1..zn, ...]
                batch_size_effective = len(batch_chunk_end_ids)
                outputs_per_input = outputs.view(batch_size_effective, num_return_sequences, -1)

                ignore_ids = {
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                    tokenizer.bos_token_id,
                }

                batch_inputs_cpu = batch_inputs["input_ids"].cpu()  # store only IDs on CPU
                del batch_inputs
                torch.cuda.empty_cache()

                for j, (chunk_end_id, inp_ids) in enumerate(zip(batch_chunk_end_ids, batch_inputs_cpu)):
                    generated = outputs_per_input[j, :, inp_ids.shape[0]:]  # strip prompt length
                    decoded_outputs = tokenizer.batch_decode(generated)
                    decoded_answers = [extract_boxed_solution(output) for output in decoded_outputs]
                    output_lengths = [sum(token_id not in ignore_ids for token_id in seq.tolist()) for seq in generated]

                    chunk_outputs.append(
                        {
                            "chunk_end_id": chunk_end_id,
                            "decoded_answers": decoded_answers,
                            "answer_output_len": output_lengths,
                        }
                    )

            chunk_methods[method] = chunk_outputs

            # clear GPU
            del outputs, generated, outputs_per_input
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        data_output.append(
            {
                'id': item['id'],
                'answer': item['answer'],
                'chunks': chunk_methods
            }
        )

        # Save `data_output` periodically
        with open(save_dir / f"chunks_answer.pkl", "wb") as f:
            pickle.dump(data_output, f)