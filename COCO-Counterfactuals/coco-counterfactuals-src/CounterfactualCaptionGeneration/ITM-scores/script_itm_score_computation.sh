#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 10

cd /export/work/tieple/Projects/COCO-Counterfactuals/coco-counterfactuals-src/CounterfactualCaptionGeneration/ITM-scores

source /home/tieple/anaconda/bin/activate bridgetower

PROMPT_PATH=$1
ORIGINAL_IMAGE_PATH=$2
COUNTER_IMAGE_PATH=$3
OUTPUT_FILENAME=$4

python ITM_score_computation.py -p ${PROMPT_PATH} -o ${ORIGINAL_IMAGE_PATH} -c ${COUNTER_IMAGE_PATH} -out ${OUTPUT_FILENAME}


