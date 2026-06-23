import os
import os.path as osp
from datetime import datetime
from pathlib import Path
import math
import subprocess

script_path = 'finetune_BT_with_vars.sh'

seed_min = 100
num_instance = 2

job_id = []
for i in range(num_instance):
    seed = seed_min + i
    data_seed = seed_min + i + 1
    cmd = ['sbatch', script_path, str(i), str(seed), str(data_seed)]
    job_id.append(subprocess.check_output(cmd).split()[-1].decode("utf-8"))