This repo was build from a fork of the [LLaVA repository](https://github.com/haotian-liu/LLaVA)

In order to train using our setup:
 - First download llava_v1_5_mix665k dataset (both text and images)
 - Place images and llava_v1_5_mix665k.json under ai.fivl/training/LLaVA/playground/data
 - Download [FiVL-Instruct Dataset](https://huggingface.co/datasets/Intel/fivl-instruct) 
 - Run fivl-instruct/inject_segments.py --output_dir <path_to_fivl_instruct>/splits --segments <path_to_fivl_instruct>/fivl_instruct.json --llava_instruct_path playground/data/llava_v1_5_mix665k.json to augment the original data with the key expressions and segmentation mask paths
 - Run training/LLaVA/scripts/v1_5/finetune_fivl.sh