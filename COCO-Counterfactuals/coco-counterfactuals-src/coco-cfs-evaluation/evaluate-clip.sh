#!/bin/bash
#SBATCH -p g48 
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --exclude=isl-gpu7
#SBATCH --qos=high

cd /export/work/tieple/Projects/COCO-Counterfactuals/coco-counterfactuals-src/coco-cfs-evaluation
source /home/tieple/anaconda/bin/activate
conda activate counterfactual

checkpoint_folder=$1
test_dataset_path=$2
output_dir=$3

python CLIP_evaluation.py \
	--checkpoint_folder ${checkpoint_folder} \
	--test_dataset_path ${test_dataset_path} \
	--output_dir ${output_dir}
