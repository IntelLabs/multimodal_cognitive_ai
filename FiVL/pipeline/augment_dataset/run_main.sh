#! /bin/bash
#SBATCH -p g48
#SBATCH --gres=gpu:1
#BATCH --ntasks-per-node=2
#BATCH --cpus-per-task=10
#SBATCH -c 10
#SBATCH -N 1
#BATCH --gres-flags=disable-binding
#SBATCH --qos=high
#SBATCH --job-name=rm_665k_gpt
#BATCH -w isl-gpu44
#SBATCH -x isl-gpu42,isl-gpu26,isl-gpu27
#SBATCH --mail-type=ALL  #BEGIN, END, FAIL,ALL
#SBATCH --mail-user=gabriela.ben.melech.stan@intel.com
#SBATCH --output=slurm-%x.%j.out

# input the data json file
JSON_FILE="/home/gbenmele/lvlm/testfiles/splits/test_data.json"
GPT_JSON_FILE="/home/gbenmele/lvlm/testfiles/gpt_splits/test_data.json"

# export https_proxy=http://proxy-chain.intel.com:911/
# export HTTPS_PROXY=http://proxy-chain.intel.com:911/
# export HTTP_PROXY=http://proxy-chain.intel.com:911/
# export http_proxy=http://proxy-chain.intel.com:911/

# Set Intel Proxies / not sure which one is the best
# export http_proxy=http://proxy-chain.intel.com:911
# export https_proxy=http://proxy-chain.intel.com:912
# export ftp_proxy=http://proxy-chain.intel.com:911
# export socks_proxy=http://proxy-chain.intel.com:1080
# export no_proxy=intel.com,.intel.com,localhost,127.0.0.1
 
# https://stackoverflow.intel.com/questions/1267
export http_proxy=http://proxy-dmz.intel.com:912
export https_proxy=http://proxy-dmz.intel.com:912
export no_proxy=.intel.com,127.0.0.1
 
# export http_proxy="http://proxy-dmz.intel.com:911"
# export https_proxy="http://proxy-dmz.intel.com:912"
# export no_proxy=".intel.com, localhost, 127.0.0.1, 192.168.0.0/16, 10.0.0.0/8"

source /home/gbenmele/miniforge3/bin/activate
conda activate fivl_data_test  #grounding_sam #
cd /home/gbenmele/lvlm/fivl_intel_repo_opensource/multimodal_cognitive_ai/FiVL/pipeline/augment_dataset

detect_model=IDEA-Research/grounding-dino-tiny
segment_model=facebook/sam-vit-huge  #facebook/sam-vit-base


IMG_DIR=/export/share/projects/mcai/LLaVA-dataset/dataset

OUT_DIR=/home/gbenmele/lvlm/testfiles/outputs


python data_main.py \
    --detector_id  ${detect_model}\
    --segmenter_id  ${segment_model}\
    --threshold 0.2 \
    --mask_union  \
    --add_second_adj \
    --should_process \
    --apply_general_mask \
    --dataset_type "llava_FT" \
    --dataset_len -1 --start 0 \
    --image_dir $IMG_DIR \
    --samples_json $JSON_FILE \
    --output_dir $OUT_DIR \
    --prompt_gpt_file ${GPT_JSON_FILE} \
    --account4realtions