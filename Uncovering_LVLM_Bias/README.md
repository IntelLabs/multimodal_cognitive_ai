# Uncovering LVLM Bias at Scale with Counterfactuals

Code from our NAACL 2025 paper:
> [Uncovering Bias in Large Vision-Language Models at Scale with Counterfactuals](https://aclanthology.org/2025.naacl-long.305/).<br> 
> Phillip Howard, Kathleen C. Fraser, Anahita Bhiwandiwalla, Svetlana Kiritchenko.<br> 
> Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics.

## Installation (tested with python 3.10.12)

1. Clone the project folder from our repository:

    ```bash
    git clone -n --depth=1 --filter=tree:0 https://github.com/IntelLabs/multimodal_cognitive_ai.git
    cd multimodal_cognitive_ai
    git sparse-checkout set --no-cone Uncovering_LVLM_Bias
    git checkout
    cd Uncovering_LVLM_Bias
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. Download and process the SocialCounterfactuals dataset:

    ```bash
    git lfs install
    git clone https://huggingface.co/datasets/Intel/SocialCounterfactuals
    python process_data.py
    ```

## Reproduce generation experiments

1. Example of how to reproduce our open-ended generation experiments for race-gender intersectional bias:

    ```bash
    mkdir output
    python lvlm_gen.py \
        --model_name llava-hf/llava-1.5-7b-hf \
        --im_path SocialCounterfactuals/images \
        --metadata_path metadata/metadata_race_gender.csv \
        --out_dir output/ \
        --batch_size 6 \
        --n_images 12 \
        --prompts keywords_characteristics
    
    python postprocess.py --out_dir output/ --metadata_path metadata/metadata_race_gender.csv
    ```

2. Evaluate toxicity of generations with the Perspective API:

    ```bash
    python perspective.py \
        --out_dir output/ \
        --occupations all \
        --prompts keywords_characteristics_0 \
        --n_samples -1 \
        --api_key <insert your Perspective API key here>
    ```

3. See evaluation.ipynb for examples showing how to calculate MaxToxicity and measure frequency of competency words.
