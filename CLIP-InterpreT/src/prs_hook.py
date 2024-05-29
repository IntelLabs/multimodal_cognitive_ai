import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import argparse
import datetime
import json
from pathlib import Path


class PRSLogger(object):
    def __init__(self, model, device):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.mlps = []
        self.post_ln_std = None
        self.post_ln_mean = None
        self.model = model
        
    @torch.no_grad()
    def compute_attentions(self, ret):
        bias_term = self.model.visual.transformer.resblocks[self.current_layer].attn.out_proj.bias
        self.current_layer += 1
        return_value = ret[:, 0].detach().cpu()
        self.attentions.append(return_value + 
                               bias_term[np.newaxis, np.newaxis, np.newaxis].cpu() / (
                                   return_value.shape[1]*return_value.shape[2])) # [b, n, h, d]
        return ret
    
    @torch.no_grad()
    def compute_mlps(self, ret):
        self.mlps.append(ret[:, 0].detach().cpu()) # [b, d]
        return ret
    
    @torch.no_grad()
    def log_post_ln_mean(self, ret):
        self.post_ln_mean = ret.detach().cpu() # [b, 1]
        return ret
        
    @torch.no_grad()
    def log_post_ln_std(self, ret):
        self.post_ln_std = ret.detach().cpu() # [b, 1] 
        return ret
    
    def _normalize_mlps(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]
        # This is just the normalization layer:
        mean_centered = (self.mlps - 
                         self.post_ln_mean[:, :, np.newaxis].to(self.device) / len_intermediates)
        weighted_mean_centered = self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[:, :, np.newaxis].to(self.device)
        bias_term = self.model.visual.ln_post.bias.detach().to(self.device) / len_intermediates
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.visual.proj.detach().to(self.device)
    
    def _normalize_attentions(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1] # 2*l + 1
        normalization_term = self.attentions.shape[2] * self.attentions.shape[3]  # n * h
        # This is just the normalization layer:
        mean_centered = (self.attentions - 
                         self.post_ln_mean[:, :, np.newaxis, np.newaxis, np.newaxis].to(self.device) / 
                         (len_intermediates * normalization_term))
        weighted_mean_centered = self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[:, :, np.newaxis, np.newaxis, np.newaxis].to(self.device)
        bias_term = self.model.visual.ln_post.bias.detach().to(self.device) / (len_intermediates * normalization_term)
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.visual.proj.detach().to(self.device)
    
    @torch.no_grad()
    def finalize(self, representation):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1).to(self.device) # [b, l, n, h, d]
        self.mlps = torch.stack(self.mlps, axis=1).to(self.device) # [b, l + 1, d]
        projected_attentions = self._normalize_attentions()
        projected_mlps = self._normalize_mlps()
        norm = representation.norm(dim=-1).detach()
        return (projected_attentions / norm[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis], 
                projected_mlps / norm[:, np.newaxis, np.newaxis])
        
    def reinit(self):
        self.current_layer = 0
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None
        torch.cuda.empty_cache()


def hook_prs_logger(model, device):
    """Hooks a projected residual stream logger to the model."""
    prs = PRSLogger(model, device)
    model.hook_manager.register('visual.transformer.resblocks.*.attn.out.post', 
                                prs.compute_attentions)

    model.hook_manager.register('visual.transformer.resblocks.*.mlp.c_proj.post', 
                                prs.compute_mlps)
    model.hook_manager.register('visual.ln_pre_post', 
                                prs.compute_mlps)
    model.hook_manager.register('visual.ln_post.mean', 
                                prs.log_post_ln_mean)
    model.hook_manager.register('visual.ln_post.sqrt_var', 
                                prs.log_post_ln_std)
    return prs
