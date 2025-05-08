#!/usr/bin/env bash

export PYTHONPATH=/home/ximinglu/neurologic-2.0
export COMMON_POS_FILENAME=/home/ximinglu/neurologic-2.0/dataset/POS/token_common_pos_100K.tsv

DATA_DIR='../dataset'
OUTPUT_FILE=$1

# gpt2
CUDA_VISIBLE_DEVICES=5 python decode_gpt2.py --model_name 'gpt2-xl' \
  --output_file ${OUTPUT_FILE} \
  --input_file ${DATA_DIR}/prefix_file.txt \
  --batch_size 8 --beam_size 20 --max_tgt_length 15 --min_tgt_length 3 \
  --ngram_size 3 --length_penalty 0.1 \
  --diversity True --sat_tolerance 2 --look_ahead 5 --num_sequences_per_return 15 \
  --do_beam_sample True --beam_temperature 10000000

# Note: since our temperature is on log likelihood level, its value needs to be large to be effective