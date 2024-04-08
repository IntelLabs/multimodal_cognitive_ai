# NeuroPrompts

*An Adaptive Framework to Optimize Prompts for Text-to-Image Generation.*

[👩‍🎨 [[Project Page](https://intellabs.github.io/multimodal_cognitive_ai/neuro_prompts/)] [[Demo](http://3.131.193.145:7860/)]  [[Video](https://youtu.be/Cmca_RWYn2g)]]

Repository for our EACL 2024 paper:

> [NeuroPrompts: An Adaptive Framework to Optimize Prompts for Text-to-Image Generation](https://arxiv.org/abs/2311.12229).<br>
> Shachar Rosenman, Vasudev Lal, and Phillip Howard.
> Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations.

## Installation

To get started, follow these steps:

1. Clone the NeuroPrompts folder from our repository:

    ```bash
    git clone -n --depth=1 --filter=tree:0 https://github.com/IntelLabs/multimodal_cognitive_ai.git
    cd multimodal_cognitive_ai
    git sparse-checkout set --no-cone NeuroPrompts
    git checkout
    cd Demos/NeuroPrompts
    ```

2. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Launching NeuroPrompts

From the NeuroPrompts folder, run the following command to launch the app:

```bash
gradio app.py
