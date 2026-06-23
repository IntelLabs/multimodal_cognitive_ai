#!/bin/bash
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --qos=low
#SBATCH --exclude=isl-gpu7

cd /export/work/tieple/Projects/COCO-Counterfactuals/coco-counterfactuals-src/Evaluation/CLIP

source /home/tieple/anaconda/bin/activate
conda activate counterfactual

CHKPT=$1
OUTDIR=$2

python CifarEvaluationBatch.py --is_10 -l -o base-clip
# python CifarEvaluationBatch.py -c ${CHKPT} --is_10 -l -o ${OUTDIR}