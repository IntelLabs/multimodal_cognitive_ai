#!/bin/bash
#SBATCH -p g80 
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --exclude=isl-gpu7
#SBATCH --qos=high

cd /export/work/tieple/Projects/COCO-Counterfactuals/coco-counterfactuals-src/Finetuning
source /home/tieple/anaconda/bin/activate
conda activate counterfactual

EXP=$1
SEED=$2
DATA_SEED=$3

# python finetune_BT.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BT-finetune-on-coco-and-coco_cfs-base-${EXP}" \
# 	--model_name "BridgeTower/bridgetower-large-itm-mlm-itc" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_base.csv" \
# 	--do_train --per_device_train_batch_size="14" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False --report_to tensorboard \
# 	--validation_file "./finetune-data/dev_coco+coco-cfs_base.csv" --do_eval --evaluation_strategy epoch \
# 	--save_strategy epoch --num_train_epochs 15 \
# 	--load_best_model_at_end=True --save_total_limit=2 --per_device_eval_batch_size="14" --seed=${SEED} --data_seed=${DATA_SEED}

# python finetune_BT.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BT-finetune-on-coco-and-coco_cfs-medium-${EXP}" \
# 	--model_name "BridgeTower/bridgetower-large-itm-mlm-itc" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_medium.csv" \
# 	--do_train --per_device_train_batch_size="14" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False --report_to tensorboard \
# 	--validation_file "./finetune-data/dev_coco+coco-cfs_medium.csv" --do_eval --evaluation_strategy epoch \
# 	--save_strategy epoch --num_train_epochs 15 \
# 	--load_best_model_at_end=True --save_total_limit=2 --per_device_eval_batch_size="14" --seed=${SEED} --data_seed=${DATA_SEED}

# python finetune_BT.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BT-finetune-on-coco-and-coco_cfs-all-${EXP}" \
# 	--model_name "BridgeTower/bridgetower-large-itm-mlm-itc" \
# 	--train_file "./finetune-data/train_coco+coco-cfs_all.csv" \
# 	--do_train --per_device_train_batch_size="14" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False --report_to tensorboard \
# 	--validation_file "./finetune-data/dev_coco+coco-cfs_all.csv" --do_eval --evaluation_strategy epoch \
# 	--save_strategy epoch --num_train_epochs 15 \
# 	--load_best_model_at_end=True --save_total_limit=2 --per_device_eval_batch_size="14" --seed=${SEED} --data_seed=${DATA_SEED}

# python finetune_BT.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BT-finetune-on-mscoco-${EXP}" \
# 	--model_name "BridgeTower/bridgetower-large-itm-mlm-itc" \
# 	--train_file "./finetune-data/train_mscoco.csv" \
# 	--do_train --per_device_train_batch_size="14" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False --report_to tensorboard \
# 	--validation_file "./finetune-data/dev_mscoco.csv" --do_eval --evaluation_strategy epoch \
# 	--save_strategy epoch --num_train_epochs 15 \
# 	--load_best_model_at_end=True --save_total_limit=2 --per_device_eval_batch_size="14" --seed=${SEED} --data_seed=${DATA_SEED}

# python finetune_BT.py \
# 	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-1/BT-finetune-on-coco-cfs-1" \
# 	--model_name "BridgeTower/bridgetower-large-itm-mlm-itc" \
# 	--train_file "./finetune-data/split-coco-cfs/train-coco-cfs.csv" \
# 	--do_train --per_device_train_batch_size="14" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
# 	--image_column image_path --caption_column caption --remove_unused_columns=False \
# 	--enforcing_counter_examples_in_same_batch=False --report_to tensorboard \
# 	--validation_file "./finetune-data/split-coco-cfs/dev-coco-cfs.csv" --do_eval --evaluation_strategy epoch \
# 	--save_strategy epoch --num_train_epochs 15 \
# 	--load_best_model_at_end=True --save_total_limit=2 --per_device_eval_batch_size="14" --seed=107 --data_seed=108

# The following failed due to 'BridgeTowerConfig' object has no attribute 'contrastive_hidden_size'
 # python finetune_BT.py \
	# --output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-1/bridgetower-large-itm-mlm-gaudi-finetune-on-coco-cfs-1" \
	# --model_name "BridgeTower/bridgetower-large-itm-mlm-gaudi" \
	# --train_file "./finetune-data/split-coco-cfs/train-coco-cfs.csv" \
	# --do_train --per_device_train_batch_size="14" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
	# --image_column image_path --caption_column caption --remove_unused_columns=False \
	# --enforcing_counter_examples_in_same_batch=False --report_to tensorboard \
	# --validation_file "./finetune-data/split-coco-cfs/dev-coco-cfs.csv" --do_eval --evaluation_strategy epoch \
	# --save_strategy epoch --num_train_epochs 15 \
	# --load_best_model_at_end=True --save_total_limit=2 --per_device_eval_batch_size="14" --seed=107 --data_seed=108

 python finetune_BT.py \
	--output_dir "/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-1/BT-finetune-on-all-original-ms-coco-1" \
	--model_name "BridgeTower/bridgetower-large-itm-mlm-itc" \
	--train_file "./finetune-data/split-coco-cfs/train-all-original-ms-coco.csv" \
	--do_train --per_device_train_batch_size="14" --learning_rate="5e-7" --warmup_steps="0" --weight_decay 0.001 \
	--image_column image_path --caption_column caption --remove_unused_columns=False \
	--enforcing_counter_examples_in_same_batch=False --report_to tensorboard \
	--validation_file "./finetune-data/split-coco-cfs/dev-all-original-ms-coco.csv" --do_eval --evaluation_strategy epoch \
	--save_strategy epoch --num_train_epochs 15 \
	--load_best_model_at_end=True --save_total_limit=2 --per_device_eval_batch_size="14" --seed=107 --data_seed=108