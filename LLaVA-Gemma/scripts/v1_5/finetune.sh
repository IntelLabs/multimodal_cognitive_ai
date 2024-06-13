#!/bin/bash
#SBATCH --job-name=gemma-ft-dev01
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=g48
#SBATCH --gres=gpu:8
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=24

# Gemma Fine-tuning dev script
# Musashi Hinck, Intel Labs

## pre-script
#cd ~/ai.llava
#. .venv/bin/activate
export WANDB_DISABLED=true
export PYTHONPATH=$PWD

model_ref=google/gemma-2b
checkpoint_path=./checkpoints/google/gemma-2b_dev01/mm_projector.bin
data_dir=./LLaVA-dataset-symlink/dataset
run_dir=${model_ref}_ft_dev01

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${model_ref} \
    --version gemma \
    --data_path $data_dir/llava_v1_5_mix665k.json \
    --image_folder ${data_dir} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${checkpoint_path} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/${run_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard
