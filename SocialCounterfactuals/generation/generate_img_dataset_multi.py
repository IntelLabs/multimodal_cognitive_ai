import argparse
import json
import sys
import os
import pickle
from pathlib import Path

import k_diffusion
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.append("./")
sys.path.append("./stable_diffusion")

from ldm.modules.attention import CrossAttention
from ldm.util import instantiate_from_config
from metrics.clip_similarity import ClipSimilarity
from contextlib import nullcontext

################################################################################
# Modified K-diffusion Euler ancestral sampler with prompt-to-prompt.
# https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def sample_euler_ancestral(model, x, sigmas, prompt2prompt_threshold=0.0, **extra_args):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        prompt_to_prompt = prompt2prompt_threshold > i / (len(sigmas) - 2)
        for m in model.modules():
            if isinstance(m, CrossAttention):
                m.prompt_to_prompt = prompt_to_prompt
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            # Make noise the same across all samples in batch.
            x = x + torch.randn_like(x[:1]) * sigma_up
    return x


################################################################################


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    print('start')
    model = instantiate_from_config(config.model)
    print('finish')
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cfg_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cfg_scale


def to_pil(image: torch.Tensor) -> Image.Image:
    image = 255.0 * rearrange(image.numpy(), "c h w -> h w c")
    image = Image.fromarray(image.astype(np.uint8))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        help="Path to output dataset directory.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=False,
        help="Path to prompts .jsonl file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Max number of images to generate per batch",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
        help="Path to stable diffusion checkpoint.",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default="stable_diffusion/models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt",
        help="Path to vae checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of sampling steps.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate per prompt (before CLIP filtering).",
    )
    parser.add_argument(
        "--max-out-samples",
        type=int,
        default=4,
        help="Max number of output samples to save per prompt (after CLIP filtering).",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=1,
        help="Number of total partitions.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        help="Partition index.",
    )
    parser.add_argument(
        "--min-p2p",
        type=float,
        default=0.1,
        help="Min prompt2prompt threshold (portion of denoising for which to fix self attention maps).",
    )
    parser.add_argument(
        "--max-p2p",
        type=float,
        default=0.9,
        help="Max prompt2prompt threshold (portion of denoising for which to fix self attention maps).",
    )
    parser.add_argument(
        "--min-cfg",
        type=float,
        default=7.5,
        help="Min classifier free guidance scale.",
    )
    parser.add_argument(
        "--max-cfg",
        type=float,
        default=15,
        help="Max classifier free guidance scale.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.2,
        help="CLIP threshold for text-image similarity of each image.",
    )
    parser.add_argument(
        "--clip-dir-threshold",
        type=float,
        default=0.2,
        help="Directional CLIP threshold for similarity of change between pairs of text and pairs of images.",
    )
    parser.add_argument(
        "--clip-img-threshold",
        type=float,
        default=0.7,
        help="CLIP threshold for image-image similarity.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast", "hmp"],
        default="hmp"
    )

    opt = parser.parse_args()
    print(opt)

    global_seed = torch.randint(1 << 32, ()).item()
    print(f"Global seed: {global_seed}")
    seed_everything(global_seed)

    model = load_model_from_config(
        OmegaConf.load("stable_diffusion/configs/stable-diffusion/v1-inference.yaml"),
        ckpt=opt.ckpt,
        vae_ckpt=opt.vae_ckpt,
    )

    device = torch.device('cuda')
    model.to(device).eval()
    
    model_wrap = k_diffusion.external.CompVisDenoiser(model)

#    clip_similarity = ClipSimilarity()

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(opt.prompts_file) as fp:
        prompts = [json.loads(line) for line in fp]

    print(f"Partition index {opt.partition} ({opt.partition + 1} / {opt.n_partitions})")
    prompts = np.array_split(list(enumerate(prompts)), opt.n_partitions)[opt.partition]

    precision_scope = torch.autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad(), precision_scope('cuda'), model.ema_scope():
        sigmas = model_wrap.get_sigmas(opt.steps)

        for i, prompt in tqdm(prompts, desc="Prompts"):
            prompt_dir = out_dir.joinpath(f"{i:07d}")
            prompt_dir.mkdir(exist_ok=True)

            with open(prompt_dir.joinpath("prompt.json"), "w") as fp:
                json.dump(prompt, fp)

            n_captions = len([i for i in prompt.keys() if i.startswith('output')]) + 1
            # n_captions = 2
            n_batches = int(np.ceil((n_captions - 1) / opt.batch_size))
            n_per_batch = int(np.ceil((n_captions - 1 + n_batches) / n_batches))
            
            uncond = model.get_learned_conditioning(n_per_batch * [""])
            cond = []
            prompt_list = []
            index_list = []
            x_out = {}
            for b in range(n_batches):
                index = list(range((b*(n_per_batch-1))+1, min((b+1)*(n_per_batch-1)+1, n_captions)))
                batch_prompts = [prompt["caption"]] + [prompt["output_" + str(i)] for i in index]
                batch_index = [0] + index
                while len(batch_prompts) < n_per_batch:
                    batch_prompts.append(prompt["caption"])
                    batch_index += [0]
                prompt_list.append(batch_prompts)
                index_list.append(batch_index)
                cond.append(model.get_learned_conditioning(batch_prompts))

            results = {}

            with tqdm(total=opt.n_samples, desc="Samples") as progress_bar:

                while len(results) < opt.n_samples:
                    seed = torch.randint(1 << 32, ()).item()
                    if seed in results:
                        continue
                    torch.manual_seed(seed)
                    x_out[seed] = []

                    x = torch.randn(1, 4, 512 // 8, 512 // 8, device=model.device) * sigmas[0]
                    x = repeat(x, "1 ... -> n ...", n=n_per_batch)

                    p2p_threshold = opt.min_p2p + torch.rand(()).item() * (opt.max_p2p - opt.min_p2p)
                    cfg_scale = opt.min_cfg + torch.rand(()).item() * (opt.max_cfg - opt.min_cfg)

                    results[seed] = dict(
                            p2p_threshold=p2p_threshold,
                            cfg_scale=cfg_scale,
                            filename=prompt["filename"],
                            caption=prompt["caption"]
                    )
                    results[seed] = {**results[seed], **{i : prompt[i] for i in prompt.keys() if i.startswith('output')}}
                   
                    for b in range(n_batches):
                        torch.manual_seed(seed)
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {"cond": cond[b], "uncond": uncond, "cfg_scale": cfg_scale}
                        samples_ddim = sample_euler_ancestral(model_wrap_cfg, x, sigmas, p2p_threshold, **extra_args)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        #x0 = x_samples_ddim[0].to('cpu')
                        
                        for k in range(x_samples_ddim.shape[0]):
                            index = index_list[b][k]
                            if index == 0 and not (b == 0 and k == 0):
                                continue
                            xk = x_samples_ddim[k].to('cpu')
                            to_pil(xk).save(prompt_dir.joinpath(f"{seed}_" + str(index) + ".jpg"), quality=100)
                            x_out[seed].append(xk)
                            #if k > 0:
                            #    
                            #    clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = clip_similarity(
                            #        x0[None].to(torch.float32), xk[None].to(torch.float32), [prompt["caption"]], [prompt_list[b][k]]
                            #    )
                            #    results[seed]['output_' + str(index) + '_clip_sim_0'] = clip_sim_0[0].item()
                            #    results[seed]['output_' + str(index) + '_clip_sim_1'] = clip_sim_1[0].item()
                            #    results[seed]['output_' + str(index) + '_clip_sim_dir'] = clip_sim_dir[0].item()
                            #    results[seed]['output_' + str(index) + '_clip_sim_image'] = clip_sim_image[0].item()

                    with open(prompt_dir.joinpath(f"metadata.jsonl"), "a") as fp:
                        fp.write(f"{json.dumps(dict(seed=seed, **results[seed]))}\n")

                    progress_bar.update()

                with open(prompt_dir.joinpath(f"images.pkl"), "wb") as output_file:
                    pickle.dump(x_out, output_file)

    print("Done.")


if __name__ == "__main__":
    main()
