#!/bin/bash
export PYTHONPATH=$PWD

# Limits internal graph size to 1000 Ops and reduces the lazy mode memory overheard.
# This will be improved in future releases. Note: This may affect performance.
export PT_HPU_MAX_COMPOUND_OP_SIZE=1000
# Sets memory pool to consume the entire HBM memory.
export PT_HPU_POOL_MEM_ACQUIRE_PERC=100

export PT_HPU_RECIPE_CACHE_CONFIG=$PWD/pt_cache,True,20000
# export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1
export LOCAL_RANK_MAP=$(hl-smi -Q module_id -f csv | tail -n +2 | tr '\n' ',' | sed 's/,$//')

MODEL_VER="${MODEL_VER:-2b-it}"
DATA_DIR="${DATA_DIR:-/data1/visual-llama}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
MODEL_PATH=google/gemma-$MODEL_VER

DEFAULT_DATA_PATH=$DATA_DIR/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
DATA_PATH="${DATA_PATH:-${DEFAULT_DATA_PATH}}"
DEFAULT_IMAGE_FOLDER=$DATA_DIR/datasets/LLaVA-Pretrain
IMAGE_FOLDER="${IMAGE_FOLDER:-${DEFAULT_IMAGE_FOLDER}}"
DEFAULT_OUTPUT_DIR=$DATA_DIR/checkpoints/llava-gemma-${MODEL_VER}-pretrain
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

DRY_RUN=${DRY_RUN:-false}

if [ -z "${DEVICE_BATCHSIZE}" ]; then
    if [ "$MODEL_VER" = "2b-it" ]; then
        DEVICE_BATCHSIZE=16
    elif [ "$MODEL_VER" = "7b-it" ]; then
        DEVICE_BATCHSIZE=8
    fi
fi

HPU_IDS=(${HABANA_VISIBLE_DEVICES//,/ })
NUM_HPU=${#HPU_IDS[@]}
GRAD_ACC=$((TOTAL_BATCH_SIZE / (DEVICE_BATCHSIZE * NUM_HPU)))
echo "Number of HPUs: $NUM_HPU"
echo "Gradient accumulation steps: $GRAD_ACC"

echo "Data path: $DATA_PATH"
echo "Image folder: $IMAGE_FOLDER"
echo "Output dir: $OUTPUT_DIR"


if [ "$DRY_RUN" = true ] ; then
    exit 0
fi

    #--image_aspect_ratio pad \
    #--group_by_modality_length True \
    # --num_train_epochs 1 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_gaudi.json \
    --model_name_or_path $MODEL_PATH \
    --version gemma \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $DEVICE_BATCHSIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --use_habana --use_lazy_mode \
    --distribution_strategy fast_ddp \
    --gaudi_config ./scripts/gaudi_config.json \
    --report_to tensorboard