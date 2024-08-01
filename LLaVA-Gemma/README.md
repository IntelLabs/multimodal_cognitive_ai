# LLaVA-Gemma GPU

Install requirements
```bash
pip install -r requirements.txt
```

slurm grab a gpu and a bunch of CPUs (64+)
```bash
pip install flash-attn --no-build-isolation
```

# LLaVA-Gemma Gaudi
Install requirements
```bash
pip install -r requirements-gaudi.txt
```


## (pre)Training
Setup your Hugging Face token
```bash
huggingface-cli login --token TOKEN
```

You may need to change data/checkpoint paths and other
variables in the:
`./scripts/v1_5/pretrain_gaudi.sh`

You can also set ENV variables to change the inputs of the script
for some but not all variables. For example,
```bash
export HABANA_VISIBLE_DEVICES=0; \
export DEVICE_BATCHSIZE=8; \
export DATA_PATH="/datasets/LLaVA-CC3M-Pretrain-595K/files/chat.json"; \
export IMAGE_FOLDER="/datasets/LLaVA-CC3M-Pretrain-595K/files/"; \
./scripts/v1_5/pretrain_gaudi.sh
```
runs pre-training on a single HPU using the 
[liuhaotian/LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)
dataset. 

You can set the `export DRY_RUN=true` variable
if you want to double check the file paths
before starting pretraining. 

## Docker

If you are using the hl-smi version `1.16` you can use the Dockerfile for gaudi to run (pre)training.
```bash
docker build -f ./docker/Dockerfile.gaudi -t llava-gemma-gaudi:latest .
```
Later, you can run the container with
```
docker run \
  -it \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --shm-size 50G \
  --cap-add=sys_nice \
  --net=host \
  llava-gemma-gaudi:latest
```
Notice that we are using `HABANA_VISIBLE_DEVICES=all` here. You might need to
further set `HABANA_VISIBLE_DEVICES` inside the container or change `./scripts/v1_5/pretrain_gaudi.sh`
depending on your needs.

# CLI usage
```bash
python -m llava.serve.cli --model-path /path/to/model/dir/ --image-file image.jpg
```