#!/bin/bash
#SBATCH -p g24 
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --exclude=isl-gpu7
#SBATCH --qos=high

cd /export/work/tieple/Projects/COCO-Counterfactuals/coco-counterfactuals-src/Finetuning
source /home/tieple/anaconda/bin/activate
conda activate counterfactual

# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-base-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_base.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \

#     --logging_steps 109 --report_to tensorboard \
# 	--validation_file "./finetune-data/dev_coco+coco-cfs_base.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108 


# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-medium-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_medium.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \
#     --logging_steps 109 --report_to tensorboard \
# 	--validation_file "./finetune-data/dev_coco+coco-cfs_medium.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108

# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-all-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_all.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \
#     --logging_steps 109 --report_to tensorboard \
# 	--validation_file "./finetune-data/dev_coco+coco-cfs_all.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108

# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-mscoco-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/train_mscoco.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \
#     --logging_steps 109 --report_to tensorboard \
# 	--validation_file "./finetune-data/dev_mscoco.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108

# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-mscoco-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/train_mscoco.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \
#     --logging_steps 109 --report_to tensorboard \
#     --validation_file "./finetune-data/splitted_dev_mscoco.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 \
#     --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108

# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-coco-and-coco_cfs-all-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_all.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \
#     --logging_steps 109 --report_to tensorboard \
#     --validation_file "./finetune-data/splitted_dev_coco+coco-cfs_all.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 \
#     --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108

# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-coco-and-coco_cfs-medium-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_medium.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \
#     --validation_file "./finetune-data/splitted_dev_coco+coco-cfs_medium.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 \
#     --logging_steps 109 --report_to tensorboard \
#     --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108

# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-coco-and-coco_cfs-base-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_base.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \
#     --logging_steps 109 --report_to tensorboard \
#     --validation_file "./finetune-data/splitted_dev_coco+coco-cfs_base.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 \
#     --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108 

# python finetune_CLIP.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-1/CLIP-finetune-on-coco-cfs-1" \
# 	--model_name "openai/clip-vit-base-patch32" \
# 	--train_file "./finetune-data/split-coco-cfs/train-coco-cfs.csv" \
# 	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False \
#     --logging_steps 109 --report_to tensorboard \
#     --validation_file "./finetune-data/split-coco-cfs/dev-coco-cfs.csv"  \
#     --do_eval --evaluation_strategy steps --eval_steps 109 \
#     --save_strategy steps --save_steps 109 \
#     --num_train_epochs 30 --load_best_model_at_end=True \
#     --save_total_limit=2 --per_device_eval_batch_size="128" \
#     --seed=107 --data_seed=108 

python finetune_CLIP.py \
	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-1/CLIP-finetune-on-all-original-ms-coco-1" \
	--model_name "openai/clip-vit-base-patch32" \
	--train_file "./finetune-data/split-coco-cfs/train-all-original-ms-coco.csv" \
	--do_train --per_device_train_batch_size="128" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
	--image_column image_path --caption_column caption --remove_unused_columns=False \
	--enforcing_counter_examples_in_same_batch=False \
    --logging_steps 109 --report_to tensorboard \
    --validation_file "./finetune-data/split-coco-cfs/dev-all-original-ms-coco.csv"  \
    --do_eval --evaluation_strategy steps --eval_steps 109 \
    --save_strategy steps --save_steps 109 \
    --num_train_epochs 30 --load_best_model_at_end=True \
    --save_total_limit=2 --per_device_eval_batch_size="128" \
    --seed=107 --data_seed=108 