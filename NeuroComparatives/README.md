# NeuroComparatives

Code from our NAACL 2024 paper:
> [NeuroComparatives: Neuro-Symbolic Distillation of Comparative Knowledge](https://aclanthology.org/2024.findings-naacl.281.pdf).<br> 
> Phillip Howard, Junlin Wang, Vasudev Lal, Gadi Singer, Yejin Choi, Swabha Swayamdipta.<br> 
> Findings of the Association for Computational Linguistics: NAACL 2024

## Installation

1. Clone the project folder from our repository:

    ```bash
    git clone -n --depth=1 --filter=tree:0 https://github.com/IntelLabs/multimodal_cognitive_ai.git
    cd multimodal_cognitive_ai
    git sparse-checkout set --no-cone NeuroComparatives
    git checkout
    cd NeuroComparatives
    ```

2. Create the conda environment:

    ```bash
    conda env create -f environment.yaml
    conda activate neurologic
    ```

3. Download gpt2-xl and flair:

    ```bash
    mkdir output
    mkdir models
    cd models
    git lfs install
    git clone https://huggingface.co/flair/pos-english
    mv pos-english flair_pos_english
    git clone https://huggingface.co/openai-community/gpt2-xl
    ```

## Generate NeuroComparatives

1. Set environment variables:

    ```bash
    cd ../Relational_Knowledge
    export COMMON_POS_FILENAME=../dataset/POS/token_common_pos_100K.tsv
    export COMPARATIVE_WORDS_FILENAME=../Relational_Knowledge/data/Comparative_Words/frequent_list.txt
    export UNCOUNTABLE_NOUNS_FILENAME=../Relational_Knowledge/data/Countable_Nouns/combined_set.pkl
    export COMPARATIVE_WORDS_FILENAME=../Relational_Knowledge/data/Comparative_Words/frequent_list.txt
    ```

2. Prepare prompts and constraints for the sample input entities file. To generate NeuroComparatives for your own entities, create a tab-delimited file in the same format where the first column is a class name and the last two columns contain the entity pair.

    ```bash
    python NeuroLogic_prompt_constraint.py \
    --input_path ../input/entities_0_to_10k_flat.csv \
    --output_file ../output/entities_0_to_10k_flat_rdy2generate.csv \
    --general_constraint_path constraints/prefix_file_general2.txt \
    --model_name ../models/gpt2-xl \
    --contrast \
    --beam_size 15 \
    --look_ahead 0 \
    --aux_num 1 \
    --entity_as_prompt \
    --divide_aux \
    --divide_adv \
    --constraint_method all_type_knowledge2 \
    --conditional compared_to3
    ```

3. Generate the NeuroComparatives:

    ```bash
    python NeuroLogic_generate.py \
    --input_path ../output/entities_0_to_10k_flat_rdy2generate.csv \
    --output_file ../output/entities_0_to_10k_generated.txt \
    --constraint_path constraints/empty.pkl \
    --general_constraint_path constraints/prefix_file_general2.txt \
    --model_name ../models/gpt2-xl \
    --batch_size 1 \
    --beam_size 15 \
    --min_tgt_length 5 \
    --max_tgt_length 32 \
    --ngram_size 3 \
    --length_penalty 0.1 \
    --sat_tolerance 3 \
    --beta 1.25 \
    --contrast \
    --diversity \
    --do_beam_sample \
    --use_general_constraints \
    --num_return_sequences 10 \
    --beam_temperature 10000000 \
    --lp_return_size 3 \
    --look_ahead 0 \
    --ordered \
    --ordered_type first_order \
    --special_comparative_constraint Comparative_Whole
    ```

4. Filter the generated comparatives:
    ```bash 
    cd eval
    python run_eval_slurm.py \
    --input_file ../../output/entities_0_to_10k_generated.txt  \
    --output_file ../../output/entities_0_to_10k_generated.csv \
    --thresh 0.5 \
    --n_cpu 14 \
    --all_hyps \
    --ordered
    ```