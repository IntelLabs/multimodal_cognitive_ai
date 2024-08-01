# ==============================================================================
# Copyright (c) [2024] [Intel Labs]
#
# Original Copyright:
# Copyright (c) [2023] [Haotian Liu]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications:
# - 2024: Adapted for training Gemma LLM with Intel Gaudi 2 AI accelerators.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# Description:
# model interation via cmd line

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


from transformers import (
  LlavaForConditionalGeneration,
  LlavaConfig,
  AutoTokenizer,
  AutoProcessor,
  AddedToken,
  CLIPImageProcessor
)

from llava.model.language_model.llava_gemma import LlavaGemmaConfig


def convert_gemma_to_llava(gemma_config):
    llava_config_dict = {
        "_name_or_path": "llava-gemma",
        "architectures": ["LlavaForConditionalGeneration"],
        "ignore_index": -100,
        "image_token_index": 256000,
        "model_type": "llava",
        "projector_hidden_act": "gelu",
        "text_config": {
            "_name_or_path": gemma_config._name_or_path,
            "architectures": ["GemmaForCausalLM"],
            "bos_token_id": gemma_config.bos_token_id,
            "eos_token_id": gemma_config.eos_token_id,
            "head_dim": gemma_config.head_dim,
            "hidden_act": "gelu_pytorch_tanh",  # Assuming we want to keep the same activation function name
            "hidden_activation": gemma_config.hidden_act,
            "hidden_size": gemma_config.hidden_size,
            "intermediate_size": gemma_config.intermediate_size,
            "max_position_embeddings": gemma_config.max_position_embeddings,
            "model_type": "gemma",
            "num_attention_heads": gemma_config.num_attention_heads,
            "num_hidden_layers": gemma_config.num_hidden_layers,
            "num_key_value_heads": gemma_config.num_key_value_heads,
            "pad_token_id": gemma_config.pad_token_id,
            "rope_scaling": gemma_config.rope_scaling,
            "tie_word_embeddings": True,  # Assuming this needs to be set to True
            "torch_dtype": gemma_config.torch_dtype,
            "vocab_size": gemma_config.vocab_size
        },
        "torch_dtype": "float32",
        "transformers_version": gemma_config.transformers_version,
        "vision_config": {
            "hidden_size": 1024,
            "image_size": 336,
            "intermediate_size": 4096,
            "model_type": "clip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "projection_dim": 768,
            "vocab_size": 32000
        },
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default"
    }

    return LlavaConfig.from_dict(llava_config_dict)

def rename_keys_in_statedict(state_dict):
    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()

    new_state_dict = {}
    for key, value in state_dict.items():
        
        if key.startswith("model.vision_tower.vision_tower."):
            new_key = key.replace("model.vision_tower.vision_tower.", "vision_tower.", 1)
        elif key.startswith("model."):
            new_key = key.replace("model.", "language_model.model.", 1)
        elif "lm_head" in key:
            new_key = key.replace("lm_head", "language_model.lm_head", 1)
        else:
            new_key = key
        
        new_state_dict[new_key] = value

    new = ["multi_modal_projector.linear_1.weight",  "multi_modal_projector.linear_1.bias",   "multi_modal_projector.linear_2.weight",  "multi_modal_projector.linear_2.bias"]
    old = ["language_model.model.mm_projector.0.weight", "language_model.model.mm_projector.0.bias", "language_model.model.mm_projector.2.weight", "language_model.model.mm_projector.2.bias"]

    for n,o in zip(new, old):
        new_state_dict[n] = new_state_dict.pop(o)

    
    return new_state_dict

def main(args):

    if args.model_base is not None:
        raise ValueError("model_base must be None. Otherwise build script will load something incorrectly.")
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)

    if 'gemma' in model_name.lower():
        conv_mode = "llava_gemma"
        model_name = "llava_gemma"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    from transformers import LlavaProcessor
    

    configgemma = LlavaGemmaConfig.from_pretrained(args.model_path + "config.json")
    config = convert_gemma_to_llava(configgemma)

    modelhf = LlavaForConditionalGeneration(config)
            
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(configgemma._name_or_path)
    tokenizer.add_tokens(["<image>"])
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(args.output_dir)

    new_sd = rename_keys_in_statedict(model.state_dict())
    modelhf.load_state_dict(new_sd)


    #weird hack to get the image token to be initialized with a normal distribution
    #https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/convert_llava_weights_to_hf.py
    pre_expansion_embeddings = modelhf.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model and pad to 64 for performance reasons
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    modelhf.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    modelhf.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (dist.sample() for _ in range(modelhf.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))
        ),
        dim=0,
    )
    modelhf.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple((dist.sample() for _ in range(modelhf.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
        dim=0,
    )

    modelhf.save_pretrained(args.output_dir)
    print("saved model to", args.output_dir)
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="my-model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)
