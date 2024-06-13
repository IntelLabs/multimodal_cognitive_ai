#!/bin/bash

export PYTHONPATH=$PWD

# Limits internal graph size to 1000 Ops and reduces the lazy mode memory overheard.
# This will be improved in future releases. Note: This may affect performance.
export PT_HPU_MAX_COMPOUND_OP_SIZE=1000
# Sets memory pool to consume the entire HBM memory.
export PT_HPU_POOL_MEM_ACQUIRE_PERC=100

export PT_HPU_RECIPE_CACHE_CONFIG=$PWD/pt_cache,True,20000
#export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1

export ENABLE_CONSOLE=true
export LOG_LEVEL=3

# Export the LOCAL_RANK_MAP environment variable
export LOCAL_RANK_MAP=$(hl-smi -Q module_id -f csv | tail -n +2 | tr '\n' ',' | sed 's/,$//')

MODEL_VER=2b-it
DATA_DIR=/mnt/weka/intellabs/LLaVA
DATA_DIR=/data1/visual-llama
TOTAL_BATCH_SIZE=128
MODEL_PATH=google/gemma-$MODEL_VER

if [ "$MODEL_VER" = "2b-it" ]; then
    DEVICE_BATCHSIZE=2
elif [ "$MODEL_VER" = "7b-it" ]; then
    DEVICE_BATCHSIZE=1
fi

HPU_IDS=(${HABANA_VISIBLE_DEVICES//,/ })
NUM_HPU=${#HPU_IDS[@]}
GRAD_ACC=$((TOTAL_BATCH_SIZE / (DEVICE_BATCHSIZE * NUM_HPU)))
echo "Number of HPUs: $NUM_HPU"
echo "Gradient accumulation steps: $GRAD_ACC"

deepspeed   llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_gaudi.json \
    --model_name_or_path $MODEL_PATH \
    --version gemma \
    --data_path $DATA_DIR/playground/data/llava_v1_5_mix665k.json \
    --image_folder $DATA_DIR/datasets \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter  /data1/checkpoints/llava-gemma-${MODEL_VER}-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /root/llava-gemma-${MODEL_VER}-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size $DEVICE_BATCHSIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --use_habana --use_lazy_mode \
    --distribution_strategy fast_ddp \
    --gaudi_config scripts/gaudi_config.json \
    --report_to tensorboard \