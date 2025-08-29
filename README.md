# Repo Structure
```
📦llm_reasoning
 ┣ 📂data
 ┃ ┣ 📂aime_1983_2023
 ┃ ┃ ┗ 📜train.jsonl
 ┃ ┣ 📂gsm8k
 ┃ ┃ ┣ 📜test.jsonl
 ┃ ┃ ┗ 📜train.jsonl
 ┃ ┣ 📂math
 ┃ ┃ ┣ 📜test.jsonl
 ┃ ┃ ┗ 📜train.jsonl
 ┃ ┗ 📂outputs                              # processed output datasets (used for visualizations)
 ┃   ┗ 📂Qwen3-8B
 ┃     ┣ 📂gsm8k
 ┃     ┃ ┣ 📜chunks_answer.pkl
 ┃     ┃ ┣ 📜chunks_answer_probs.pkl
 ┃     ┃ ┗ 📜cots.pkl
 ┃     ┗ 📂math
 ┃       ┗ 📂Algebra
 ┃         ┣ 📂level_1
 ┃         ┃ ┣ 📜chunks_answer.pkl
 ┃         ┃ ┣ 📜chunks_answer_probs.pkl
 ┃         ┃ ┗ 📜cots.pkl
 ┃        ...
 ┃         ┗ 📂level_5
 ┃           ┣ 📜chunks_answer.pkl
 ┃           ┣ 📜chunks_answer_probs.pkl
 ┃           ┗ 📜cots.pkl
 ┣ 📂notebooks
 ┃ ┗ 📜test.ipynb                           # Contains the code used to develop the scripts, and all visualizations
 ┣ 📂src
 ┃ ┣ 📜generate_answer.py                   # Input: aime_1983_2023/gsm8k/math train set; Output: {dataset_path}/cots.pkl
 ┃ ┣ 📜generate_answer_probs.py             # Input: {dataset_path}/cots.pkl; Output: {dataset_path}/chunks_answer_probs.pkl
 ┃ ┗ 📜generate_cot.py                      # Input: {dataset_path}/cots.pkl; Output: {dataset_path}/chunks_answer.pkl
 ┗ 📜README.md
 ```