#!/usr/bin/env bash

# 2) Prepare a JSONL (or JSON) dataset:
# each line: {"prompt": "...", "question": "...", "answer": "..."}

# 3) Train the compressor for a Qwen decoder:
#python train.py \
#  --train_path data/pqa.jsonl \
#  --decoder Qwen/Qwen2.5-0.5B-Instruct \
#  --compressor xlm-roberta-xl \
#  --save_dir ckpt_compressor_qwen_xlmr \
#  --epochs 1 --batch_size 2 --lr 5e-6 --lambda_length 0.5 --entropy_beta 1e-3

python train.py \
  --train_path "databricks/databricks-dolly-15k" \
  --decoder "Qwen/Qwen3-0.6B" \
  --compressor "facebook/xlm-roberta-xl" \
  --save_dir "ckpt_compressor_dolly" \
  --epochs 1 --batch_size 2 --lr 5e-6 --lambda_length 0.5 --entropy_beta 1e-3


## 4) Inference (example sketch inside Python):
#from model import PromptCompressionWrapper
#m = PromptCompressionWrapper("Qwen/Qwen2.5-0.5B-Instruct", "xlm-roberta-xl")
## optionally load your trained compressor weights:
## state = torch.load("ckpt_compressor_qwen_xlmr/epoch_1/compressor_state.pt", map_location="cpu")
## m.compressor.load_state_dict(state)
#ans = m.generate(prompt_text="(long prompt...)", question_text="What is ...?", threshold=0.6, max_new_tokens=128)
#print(ans)

