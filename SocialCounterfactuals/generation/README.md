# Counterfactual image generation

The following instructions provide details on how to generate SocialCounterfactuals (tested with python 3.10.12):

1. Create a virtual environment and install dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

2. Clone the Instruct-Pix2Pix repo and copy our modified files into it:

    ```bash
    git clone https://github.com/timothybrooks/instruct-pix2pix.git
    cp generate_img_dataset_multi.py instruct-pix2pix/dataset_creation/
    cp attention.py instruct-pix2pix/stable_diffusion/ldm/modules/
    ```

3. Create an output directory where the images will be saved and download stable diffusion weights:

    ```bash
    mkdir output
    cd instruct-pix2pix/
    bash scripts/download_pretrained_sd.sh
    ```

4. Generate SocialCounterfactuals for race-gender intersectional social attributes:

    ```bash
    python dataset_creation/generate_img_dataset_multi.py \
        --out_dir ../output \
        --prompts_file ../captions/race_gender_occupation.jsonl \
        --batch_size 2
    ```

To generate SocialCounterfactuals for other intersectional attributes, modify the `prompts_file` argument in step 4 to point to the correspnding file in the `captions` subdirectory.