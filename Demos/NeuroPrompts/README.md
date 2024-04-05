# NeuroPrompts

Repository for our EACL 2024 paper:

> [NeuroPrompts: An Adaptive Framework to Optimize Prompts for Text-to-Image Generation](https://arxiv.org/abs/2311.12229).<br>
> Shachar Rosenman, Vasudev Lal, and Phillip Howard.
> Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations.

## Installation

First, clone the NeuroPrompts folder from our repository:
```
git clone -n --depth=1 --filter=tree:0 https://github.com/IntelLabs/multimodal_cognitive_ai.git
cd multimodal_cognitive_ai
git sparse-checkout set --no-cone NeuroPrompts
git checkout
cd Demos/NeuroPrompts
```

Install the required packages with pip:
```
pip install -r requirements.txt
```

## Launching NeuroPrompts

From the NeuroPrompts folder, run the following command to launch the app:
```
gradio app.py
```
