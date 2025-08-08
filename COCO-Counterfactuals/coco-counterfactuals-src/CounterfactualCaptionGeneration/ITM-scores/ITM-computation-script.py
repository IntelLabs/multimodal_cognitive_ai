import os
import os.path as osp
from datetime import datetime
from pathlib import Path
import math
import subprocess

script_path = 'script_itm_score_computation.sh'
n_gpu = 30
n_per_gpu = 1

prompt_file_path = '../output/gen_captions/perplexity_score/sim_.8_.91/cap_val_2017.txt'
original_image_path = "../../../coco-counterfactuals/synthetic-images-for-original-captions"
counterfactual_image_path = '../../../coco-counterfactuals/synthetic-images-for-counterfactual-captions'
output_path = 'output-ITM-scores'
output_prefix = 'scores'
output_filetype = '.csv'
#laod lines
lines = []
with open(prompt_file_path, 'r') as r:
    lines = [line.rstrip() for line in r]
    
intermediate_path = "tmp_itm_" + str(n_gpu)
Path(intermediate_path).mkdir(parents=True, exist_ok=True)

n_split = n_gpu * n_per_gpu
n_lines = int(math.ceil(len(lines) / n_split))

file_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
input_prefix = 'captions'
start = 0
for i in range(n_split):
    data_split_path = os.path.join(intermediate_path, input_prefix + '_' + file_ts + '_' + str(i) + '.txt')
    with open(data_split_path, 'w') as fo:
        end = ((i+1)*n_lines)
        for line in lines[start:end]:
            fo.write(line + '\n')
        start = end
    if end >= len(lines):
        n_split = i+1
        break

#create output folder 
Path(output_path).mkdir(parents=True, exist_ok=True)

job_id = []
for i in range(int(math.ceil(n_split/n_per_gpu))):
    s = i*n_per_gpu
    cmd = ['sbatch', script_path]
    for j in range(n_per_gpu):
        data_split_path = os.path.join(intermediate_path, input_prefix + '_' + file_ts + '_' + str(s+j) + '.txt')
        output_filename = output_prefix + str(s+j) + output_filetype
        cmd += [data_split_path]
        cmd += [original_image_path, counterfactual_image_path, os.path.join(output_path, output_filename)]
    job_id.append(subprocess.check_output(cmd).split()[-1].decode("utf-8"))