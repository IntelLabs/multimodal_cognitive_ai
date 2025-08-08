#!/bin/bash

sbatch Caltech_slurm.sh . base-bridgetower

for c in "BT-finetune-on-mscoco-0" "BT-finetune-on-coco-and-coco_cfs-all-1/checkpoint-20895" "BT-finetune-on-coco-and-coco_cfs-base-0" "BT-finetune-on-coco-and-coco_cfs-medium-0/checkpoint-19904"
do
    sbatch Caltech_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/BridgeTower/${c} ${c}
done
# sbatch Cifar10_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-base-1 CLIP-finetune-on-coco-and-coco_cfs-base-1
# sbatch Cifar10_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-medium-1 CLIP-finetune-on-coco-and-coco_cfs-medium-1
# sbatch Cifar10_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-all-1 CLIP-finetune-on-coco-and-coco_cfs-all-1
# sbatch Cifar10_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-mscoco-1 CLIP-finetune-on-mscoco-1

# sbatch Cifar100_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-base-1 CLIP-finetune-on-coco-and-coco_cfs-base-1
# sbatch Cifar100_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-medium-1 CLIP-finetune-on-coco-and-coco_cfs-medium-1
# sbatch Cifar100_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-all-1 CLIP-finetune-on-coco-and-coco_cfs-all-1
# sbatch Cifar100_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-mscoco-1 CLIP-finetune-on-mscoco-1

# sbatch Caltech_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-base-1 CLIP-finetune-on-coco-and-coco_cfs-base-1
# sbatch Caltech_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-medium-1 CLIP-finetune-on-coco-and-coco_cfs-medium-1
# sbatch Caltech_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-all-1 CLIP-finetune-on-coco-and-coco_cfs-all-1
# sbatch Caltech_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-mscoco-1 CLIP-finetune-on-mscoco-1

# sbatch Food101_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-base-1 CLIP-finetune-on-coco-and-coco_cfs-base-1
# sbatch Food101_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-medium-1 CLIP-finetune-on-coco-and-coco_cfs-medium-1
# sbatch Food101_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-all-1 CLIP-finetune-on-coco-and-coco_cfs-all-1
# sbatch Food101_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-mscoco-1 CLIP-finetune-on-mscoco-1

# sbatch ImageNet_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-base-1 CLIP-finetune-on-coco-and-coco_cfs-base-1
# sbatch ImageNet_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-medium-1 CLIP-finetune-on-coco-and-coco_cfs-medium-1
# sbatch ImageNet_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-coco-and-coco_cfs-all-1 CLIP-finetune-on-coco-and-coco_cfs-all-1
# sbatch ImageNet_slurm.sh /export/share/projects/mcai/COCO-Counterfactuals/checkpoints/CLIP/CLIP-finetune-on-mscoco-1 CLIP-finetune-on-mscoco-1