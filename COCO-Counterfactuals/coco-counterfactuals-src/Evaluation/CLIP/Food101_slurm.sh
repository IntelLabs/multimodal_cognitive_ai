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

# python Food101EvaluationBatch.py -l -o base-clip
python Food101EvaluationBatch.py -c ${CHKPT} -l -o ${OUTDIR}