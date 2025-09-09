#!/usr/bin/env bash

export HF_HOME=$WORK/hf_cache

python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Algebra --level 1
python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Algebra --level 2
python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Algebra --level 3
python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Algebra --level 4
python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Algebra --level 5

python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Prealgebra --level 1
python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Prealgebra --level 2
python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Prealgebra --level 3
python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Prealgebra --level 4
python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset math --subject Prealgebra --level 5

python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset aime

python generate_cot_alliot_local.py --model_name "Qwen/Qwen3-8B" --dataset gsm8k

