# LLaVA-Gemma GPU

Install requirements
`pip install -r requirements.txt`

slurm grab a gpu and a bunch of CPUs (64+)
`pip install flash-attn --no-build-isolation`

# LLaVA-Gemma Gaudi
Install requirements
`pip install -r requirements-gaudi.txt`

you might need to comment out the last line of requirements_gaudi.txt (transformers) and pip install that after
`pip install transformers@git+https://github.com/huggingface/transformers@49204c1d37b807def930fe45f5f84abc370a7200`

# (pre)Training
You may need to change your data/checkpoint paths:
`./scripts/v1_5/pretrain_gaudi.sh`

# CLI usage
`python -m llava.serve.cli --model-path /path/to/model/dir/ --image-file image.jpg`
