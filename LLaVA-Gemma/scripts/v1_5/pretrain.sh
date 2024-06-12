#!/bin/bash
#SBATCH --job-name=gemma-gpu-dev01
#SBATCH --output=./logs/slurm-%A.%a.out
#SBATCH --error=./logs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=g48
#SBATCH --gres=gpu:4
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=24


## pre-script
#cd ~/ai.llava-gemma
#. .venv/bin/activate
#conda activate llavagemma
export WANDB_DISABLED=true
export PYTHONPATH=$PWD

model_ref=google/gemma-2b
run_dir=${model_ref}_dev01
DATA_DIR=./LLaVA-dataset-symlink/dataset/LLaVA-Pretrain
IMG_DIR=./LLaVA-dataset-symlink/dataset/LLaVA-Pretrain/images

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${model_ref} \
    --version gemma \
    --data_path $DATA_DIR/blip_laion_cc_sbu_558k.json \
    --image_folder $IMG_DIR \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter true \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end false \
    --mm_use_im_patch_token false \
    --bf16 true \
    --output_dir ./checkpoints/${run_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 true \
    --model_max_length 2048 \
    --gradient_checkpointing true \
    --dataloader_num_workers 16 \
    --lazy_preprocess true
    # --report_to wandb
