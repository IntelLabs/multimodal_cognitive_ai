#!/bin/bash
#SBATCH -p g80 
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --qos=low
#SBATCH --exclude=isl-gpu7

cd /export/work/tieple/Projects/COCO-Counterfactuals/coco-counterfactuals-src/Evaluation/BridgeTower/itr-Flickr30K

source /home/tieple/anaconda/bin/activate
conda activate counterfactual

CHKPT=$1
OUTDIR=$2

python BT-evaluation-batch-on-Flickr30K.py -c ${CHKPT} -o ${OUTDIR}