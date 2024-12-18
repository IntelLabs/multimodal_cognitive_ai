
In order to evaluate our dataset using GPT4o-as-a-judge:
 - First download llava_v1_5_mix665k dataset (both text and images)
 - Place images and llava_v1_5_mix665k.json under ai.fivl/training/LLaVA/playground/data
 - Download [FiVL-Instruct Dataset](https://huggingface.co/datasets/Intel/fivl-instruct) 
 - Run fivl-instruct/inject_segments.py --output_dir <path_to_fivl_instruct>/splits --segments <path_to_fivl_instruct>/fivl_instruct.json --llava_instruct_path playground/data/llava_v1_5_mix665k.json to augment the original data with the key expressions and segmentation mask paths
 - eval_keyword.py generates the GPT4o scores for the key expressions automatic evaluation
 - eval_seg1.py generates the GPT4o outputs for the Seg1 setup.
 - eval_seg2.py generates the GPT4o outputs for the Seg2 setup.