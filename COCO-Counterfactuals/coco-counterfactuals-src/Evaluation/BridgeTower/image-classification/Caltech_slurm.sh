#!/bin/bash
#SBATCH -p g80 
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --qos=low
#SBATCH --exclude=isl-gpu7

cd /export/work/tieple/Projects/COCO-Counterfactuals/coco-counterfactuals-src/BridgeTower/image-classification

source /home/tieple/anaconda/bin/activate
conda activate counterfactual

CHKPT=$1
OUTDIR=$2

# python CaltechEvaluationBatch.py -l -o base-clip
# python Caltech256EvaluationBatch.py -l -o base-clip
# python CaltechEvaluationBatch.py -c ${CHKPT} -l -o ${OUTDIR}
python CaltechEvaluationBatch.py -c ${CHKPT} -l -o ${OUTDIR}