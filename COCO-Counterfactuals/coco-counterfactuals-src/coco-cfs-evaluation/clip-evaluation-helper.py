import os
import os.path as osp
from datetime import datetime
from pathlib import Path
import math
import subprocess

script_path = 'evaluate-clip.sh'

configs = []

# configs.append({
#     'checkpoint_folder' : 'baseline',
#     'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_all.csv',
#     'output_dir' : 'splitted_test_coco+coco-cfs_all/base-model'
# })

configs.append({
    'checkpoint_folder' : '/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-mscoco-1',
    'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_all.csv',
    'output_dir' : 'splitted_test_coco+coco-cfs_all/CLIP-finetune-on-mscoco-1'
})

configs.append({
    'checkpoint_folder' : '/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-coco-and-coco_cfs-all-1',
    'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_all.csv',
    'output_dir' : 'splitted_test_coco+coco-cfs_all/CLIP-finetune-on-coco-and-coco_cfs-all-1'
})


configs.append({
    'checkpoint_folder' : 'baseline',
    'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_base.csv',
    'output_dir' : 'splitted_test_coco+coco-cfs_base/base-model'
})

configs.append({
    'checkpoint_folder' : '/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-mscoco-1',
    'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_base.csv',
    'output_dir' : 'splitted_test_coco+coco-cfs_base/CLIP-finetune-on-mscoco-1'
})

configs.append({
    'checkpoint_folder' : '/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-coco-and-coco_cfs-base-1',
    'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_base.csv',
    'output_dir' : 'splitted_test_coco+coco-cfs_base/CLIP-finetune-on-coco-and-coco_cfs-base-1'
})


configs.append({
    'checkpoint_folder' : 'baseline',
    'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_medium.csv',
    'output_dir' : 'splitted_test_coco+coco-cfs_medium/base-model'
})

configs.append({
    'checkpoint_folder' : '/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-mscoco-1',
    'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_medium.csv',
    'output_dir' : 'splitted_test_coco+coco-cfs_medium/CLIP-finetune-on-mscoco-1'
})

configs.append({
    'checkpoint_folder' : '/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-nodevset/CLIP/CLIP-finetune-on-coco-and-coco_cfs-medium-1',
    'test_dataset_path' : '../Finetuning/finetune-data/splitted_test_coco+coco-cfs_medium.csv',
    'output_dir' : 'splitted_test_coco+coco-cfs_medium/CLIP-finetune-on-coco-and-coco_cfs-medium-1'
})



configs.append({
    'checkpoint_folder' : 'baseline',
    'test_dataset_path' : '../Finetuning/finetune-data/split-coco-cfs/test-coco-cfs.csv',
    'output_dir' : 'test-coco-cfs/base-model'
})

configs.append({
    'checkpoint_folder' : '/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-1/CLIP-finetune-on-all-original-ms-coco-1',
    'test_dataset_path' : '../Finetuning/finetune-data/split-coco-cfs/test-coco-cfs.csv',
    'output_dir' : 'test-coco-cfs/CLIP-finetune-on-all-original-ms-coco-1'
})

configs.append({
    'checkpoint_folder' : '/export/share/projects/mcai/COCO-Counterfactuals/checkpoints-1/CLIP-finetune-on-coco-cfs-1',
    'test_dataset_path' : '../Finetuning/finetune-data/split-coco-cfs/test-coco-cfs.csv',
    'output_dir' : 'test-coco-cfs/CLIP-finetune-on-coco-cfs-1'
})


job_id = []
for config in configs:
    cmd = ['sbatch', script_path, config['checkpoint_folder'], config['test_dataset_path'], config['output_dir']]
    job_id.append(subprocess.check_output(cmd).split()[-1].decode("utf-8"))