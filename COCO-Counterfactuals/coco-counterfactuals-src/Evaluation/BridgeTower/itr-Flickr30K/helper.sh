#!/bin/bash

sbatch Flickr30_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BridgeTower/BT-finetune-on-mscoco-0 BT-finetune-on-mscoco-0
sbatch Flickr30_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BridgeTower/BT-finetune-on-coco-and-coco_cfs-all-1/checkpoint-20895 BT-finetune-on-coco-and-coco_cfs-all
sbatch Flickr30_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BridgeTower/BT-finetune-on-coco-and-coco_cfs-base-0 BT-finetune-on-coco-and-coco_cfs-base-0
sbatch Flickr30_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BridgeTower/BT-finetune-on-coco-and-coco_cfs-medium-0/checkpoint-19904 BT-finetune-on-coco-and-coco_cfs-medium-0
