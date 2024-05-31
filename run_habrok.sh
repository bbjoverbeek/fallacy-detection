#!/bin/bash

# manually call using `bash setup_habrok.sh`
# if job submitted, keep track using the watch command:
# watch -n 5 jobinfo 9999999 // -n x seconds refresh interval

#SBATCH --ntasks=1 
#SBATCH --time=40:00:00
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=36GB

export HF_HUB_CACHE='/scratch/s4415213/'

module purge
module load Python/3.10.8-GCCcore-12.2.0

if [ ! -d env ]; then
  python3 -m venv env
fi

source env/bin/activate
pip3 install -r requirements.txt

# model choice: uncomment the model you want to run
MODEL='meta-llama/Meta-Llama-3-8B-Instruct'
# MODEL='google/flan-t5-xl'
# MODEL='mosaicml/mpt-7b-instruct'

# possible prompt features
# PromptFeature = Literal['zero-shot', 'few-shot', 'chain-of-thought', 'self-consistency', 'positive-feedback', 'negative-feedback']

levels=( 0 1 2 )
for level in "${levels[@]}"
do
    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features zero-shot \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features few-shot \
        --n-shot 1 \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features few-shot \
        --n-shot 3 \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features few-shot \
        --n-shot 5 \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features chain-of-thought \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features self-consistency \
        --repeat 3 \
        --do-sample \
        --temp 0.7 \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features self-consistency \
        --repeat 5 \
        --do-sample \
        --temp 0.7 \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features zero-shot positive-feedback\
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features zero-shot negative-feedback\
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features self-consistency positive-feedback \
        --repeat 5 \
        --do-sample \
        --temp 0.7 \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/

    python3 prompt_model.py \
        --model $MODEL \
        --prompt-features self-consistency negative-feedback \
        --repeat 5 \
        --do-sample \
        --temp 0.7 \
        --classification_level $level \
        --dataset ./data/test/mafalda_gold_standard_dataset.jsonl \
        --logpath output/
done
