#!/usr/bin/env bash

export PYTHONPATH=/home/chandrab/github/neurologic-2.0/
export COMMON_POS_FILENAME=~/data/common_files/token_common_pos.tsv

INPUT_FILE=$1
OUTPUT_FILE=$2

# gpt2
CUDA_VISIBLE_DEVICES=0 python zero_shot/decode_gpt2.py --model_name 'gpt2-xl' \
  --output_file ${OUTPUT_FILE} \
  --input_file ${INPUT_FILE} \
  --batch_size 8 --beam_size 10 --max_tgt_length 15 --min_tgt_length 2 \
  --ngram_size 3 --length_penalty 0.1 \
  --sat_tolerance 2 --look_ahead 5 --num_sequences_per_return 10 \
  --do_beam_sample True --beam_temperature 10000000

# Note: since our temperature is on log likelihood level, its value needs to be large to be effective