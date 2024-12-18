#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from torch.nn import CrossEntropyLoss

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        vision_labels:  Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        batch_size, input_len = input_ids.shape
        non_zero_idx = torch.nonzero(input_ids == -200, as_tuple=False)
        if non_zero_idx.numel() > 0:
            img_idx = non_zero_idx[0][1]
            sample_with_img_idx =  torch.tensor([idx for idx in range(batch_size) if idx in non_zero_idx[:, 0]]).to(input_ids.device)
        else:
            img_idx = None
        
       
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        llama_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if self.config.fivl_training:
            if img_idx is not None:
                num_img_patches = int((self.model.vision_tower.config.image_size / self.model.vision_tower.config.patch_size)**2)
                vision_logits = llama_outputs.logits[:,img_idx:img_idx+num_img_patches]
                if len(sample_with_img_idx)>0:
                    vision_logits =  torch.index_select(vision_logits, dim=0, index=sample_with_img_idx) 
                    vision_labels =  torch.index_select(vision_labels, dim=0, index=sample_with_img_idx)  
                    llama_outputs = self.update_loss_with_vision_modeling(llama_outputs, vision_labels, vision_logits)
        return llama_outputs


    def update_loss_with_vision_modeling(self, llama_outputs, vision_labels, vision_logits):
        if (vision_labels==-100).all():
            return llama_outputs
        # elif vision_logits.shape[1] != vision_labels.shape[1]:
        #     return llama_outputs
        vision_labels = vision_labels.contiguous().view(-1)
        vision_logits = vision_logits.contiguous().view(-1, self.config.vocab_size)
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        vision_loss = loss_fct(vision_logits, vision_labels)
        if vision_loss <0 or vision_loss.isnan():
            print(f"vision_loss is negative: {vision_loss}")
            print(f"vision_labels range: {vision_labels.min()}, {vision_labels.max()}")
            vision_loss = llama_outputs["loss"]
        llama_outputs["loss"] =(1- self.config.lambda_seg)*llama_outputs["loss"] + self.config.lambda_seg *vision_loss
        return llama_outputs


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        fivl_training = kwargs.pop("fivl_training", False)
        lambda_seg = kwargs.pop("lambda_seg", 0.0)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.config.lambda_seg = lambda_seg
        model.config.fivl_training = fivl_training
        return model

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
