import numpy as np
from collections import OrderedDict
import torch
from torchvision import transforms
from PIL import Image
import open_clip

import slip.slip_models as models
import slip.slip_utils as utils
from slip.slip_tokenizer import SimpleTokenizer

from transformers import (
    CLIPImageProcessor, BlipImageProcessor, XCLIPProcessor, EfficientNetImageProcessor,
    CLIPTokenizerFast, BertTokenizerFast, CLIPTokenizer,
    CLIPModel, BlipModel, FlavaModel, XCLIPModel, Blip2Model, AlignModel,
    Pix2StructTextModel, FlavaTextModel,
    Pix2StructVisionModel, FlavaImageModel,
    FlavaMultimodalModel, FlavaImageProcessor,
    BridgeTowerImageProcessor, RobertaTokenizerFast, BridgeTowerModel,
    Blip2Processor, Blip2ForConditionalGeneration, BlipForConditionalGeneration,
    Pix2StructForConditionalGeneration
)
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from alip import (
    create_model, tokenize, factory
)

class CLIP_zeroshot(torch.nn.Module):
    def __init__(self, num_frames=1, input_res=224, model_name="openai/clip-vit-base-patch32", device='cuda'):
        super(CLIP_zeroshot, self).__init__()
        self.num_frames = num_frames
        self.input_res = input_res
        self.device = device

        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.custom_transforms = transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_visual_features(self, visual_input, is_custom_transforms=False):
        if(not is_custom_transforms):
            processed_visual_input = self.clip_image_processor(visual_input, size=self.input_res)
            pixel_values = torch.tensor(np.array(processed_visual_input['pixel_values'])).to(self.device)
        else:
            pixel_values = self.custom_transforms(visual_input)
            pixel_values = pixel_values.reshape(-1, 3, self.input_res, self.input_res)
        visual_features = self.clip.get_image_features(pixel_values)
        visual_features = visual_features.reshape(1, self.num_frames, -1)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
        visual_features = torch.mean(visual_features, 1)
        return visual_features

    def get_text_features(self, text_input):
        tokenizer_outputs = self.tokenizer(text_input, truncation=True, padding=True, return_tensors="pt")
        tokenizer_outputs = {k: torch.LongTensor(np.array(v)).to(self.device) for k, v in tokenizer_outputs.items()}
        text_features = self.clip.get_text_features(**tokenizer_outputs)
        text_features = torch.mean(text_features, dim=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(self):
        pass

class SLIP_zeroshot(torch.nn.Module):
    def __init__(self, num_frames=1, input_res=224, model_name="openai/clip-vit-base-patch32", device='cuda'):
        super(SLIP_zeroshot, self).__init__()
        self.num_frames = num_frames
        self.input_res = input_res
        self.device = device

        ckpt = torch.load('weights/slip_base_100ep.pt')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        
        # create model
        old_args = ckpt['args']
        print("=> creating model: {}".format(old_args.model))
        self.model = getattr(models, old_args.model)(rand_embed=False,
            ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        self.model.to(self.device)
        self.model.load_state_dict(state_dict, strict=True)

        self.tokenizer = SimpleTokenizer()

        self.custom_transforms = transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_visual_features(self, visual_input, is_custom_transforms=False):
        #visual_input = inputs.detach().cpu()
        #print(visual_input.size())
        pixel_values = self.custom_transforms(visual_input)
        pixel_values = pixel_values.reshape(-1, 3, self.input_res, self.input_res)
        visual_features = utils.get_model(self.model).encode_image(pixel_values)
        visual_features = visual_features.reshape(1, self.num_frames, -1)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
        visual_features = torch.mean(visual_features, 1)
        return visual_features

    def get_text_features(self, text_input):
        tokenizer_outputs = self.tokenizer(text_input).to(self.device)
        tokenizer_outputs = tokenizer_outputs.view(-1, 77).contiguous()
        text_features = utils.get_model(self.model).encode_text(tokenizer_outputs)
        text_features = torch.mean(text_features, dim=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(self):
        pass

def alip_get_state_dict(model_weight):
    state_dict = torch.load(model_weight)
    state_dict_removed = {}
    for k, value in state_dict.items():
        if "module." in k:
            k_removed = k.split("module.")[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k] = value
    return state_dict_removed

class ALIP_zeroshot(torch.nn.Module):
    def __init__(self, num_frames=1, input_res=224, model_name="openai/clip-vit-base-patch32", device='cuda'):
        super(ALIP_zeroshot, self).__init__()
        self.num_frames = num_frames
        self.input_res = input_res
        self.device = device

        self.alip_model = create_model('ViT-B/32')
        state_dict = alip_get_state_dict('weights/ALIP_YFCC15M_B32.pt')
        self.alip_model.load_state_dict(state_dict, strict=True)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.alip_transforms = transforms.Compose([
            transforms.Resize(self.input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.input_res),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def get_visual_features(self, visual_input, is_custom_transforms=False):
        pixel_values = self.alip_transforms(visual_input)
        pixel_values = pixel_values.reshape(-1, 3, self.input_res, self.input_res)
        visual_features = self.alip_model.encode_image(pixel_values)
        visual_features = visual_features.reshape(1, self.num_frames, -1)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
        visual_features = torch.mean(visual_features, 1)
        return visual_features

    def get_text_features(self, text_input):
        tokenizer_outputs = self.tokenizer(text_input).to(self.device)
        text_features = self.alip_model.encode_text(tokenizer_outputs)
        text_features = torch.mean(text_features, dim=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self):
        pass

class FLAVA_zeroshot(torch.nn.Module):
    def __init__(self, num_frames=1, input_res=224, model_name="openai/clip-vit-base-patch32", device='cuda'):
        super(FLAVA_zeroshot, self).__init__()
        self.num_frames = num_frames
        self.input_res = input_res
        self.device = device

        self.model_name = 'weights/flava-full/'
        self.flava_image_model = FlavaImageModel.from_pretrained(self.model_name)
        self.flava_text_model = FlavaTextModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.flava_image_processor = FlavaImageProcessor(size=self.input_res)
        self.custom_transforms = transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_visual_features(self, visual_input, is_custom_transforms=False):
        processed_visual_input = self.flava_image_processor(visual_input)
        pixel_values = torch.tensor(np.array(processed_visual_input['pixel_values'])).to(self.device)

        pixel_values = pixel_values.reshape(-1, 3, self.input_res, self.input_res)
        output = self.flava_image_model(pixel_values)
        visual_features = output.pooler_output
        visual_features = visual_features.reshape(1, self.num_frames, -1)
        visual_features = torch.mean(visual_features, 1)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
        return visual_features

    def get_text_features(self, text_input):
        tokenizer_outputs = self.tokenizer(text_input, truncation=True, padding=True, return_tensors="pt")
        tokenizer_outputs = {k: torch.LongTensor(np.array(v)).to(self.device) for k, v in tokenizer_outputs.items()}
        text_features = self.flava_text_model(**tokenizer_outputs).pooler_output
        text_features = torch.mean(text_features, dim=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

class LaCLIP_zeroshot(torch.nn.Module):
    def __init__(self, num_frames=1, input_res=224, model_name="laion400m_laclip", device='cuda'):
        super(LaCLIP_zeroshot, self).__init__()
        self.num_frames = num_frames
        self.input_res = input_res
        self.device = device

        ckpt = torch.load('weights/' + model_name + '.pt')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        
        laclip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32',
        '',
        precision='amp',
        device='cuda',
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=224,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
    )   
        self.laclip_model = laclip_model
        self.laclip_model.to(self.device)
        self.laclip_model.load_state_dict(state_dict, strict=True)

        self.laclip_text_tokenizer = SimpleTokenizer()
        self.laclip_transforms = transforms.Compose([
            transforms.Resize(self.input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.input_res),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    def get_visual_features(self, visual_input, is_custom_transforms=False):
        pixel_values = self.laclip_transforms(visual_input)
        pixel_values = pixel_values.reshape(-1, 3, self.input_res, self.input_res)
        visual_features = self.laclip_model.encode_image(pixel_values)
        visual_features = visual_features.reshape(1, self.num_frames, -1)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
        visual_features = torch.mean(visual_features, 1)
        return visual_features

    def get_text_features(self, text_input):
        tokenizer_outputs = self.laclip_text_tokenizer(text_input).to(self.device)
        tokenizer_outputs = tokenizer_outputs.view(-1, 77).contiguous()
        text_features = self.laclip_model.encode_text(tokenizer_outputs)
        text_features = torch.mean(text_features, dim=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(self):
        pass

class OpenCLIP_zeroshot(torch.nn.Module):
    def __init__(self, num_frames=1, input_res=224, model_name="ViT-B-32", device='cuda'):
        super(OpenCLIP_zeroshot, self).__init__()
        self.num_frames = num_frames
        self.input_res = input_res
        self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b79k')
        self.openclip_model = model
        self.clip_image_processor = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.openclip_transforms = transforms.Compose([
            transforms.Resize(self.input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.input_res),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def get_visual_features(self, visual_input, is_custom_transforms=False):
        pixel_values = self.openclip_transforms(visual_input)
        pixel_values = pixel_values.reshape(-1, 3, self.input_res, self.input_res)
        visual_features = self.openclip_model.encode_image(pixel_values)
        visual_features = visual_features.reshape(1, self.num_frames, -1)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
        visual_features = torch.mean(visual_features, 1)
        return visual_features

    def get_text_features(self, text_input):
        tokenizer_outputs = self.tokenizer(text_input).to(self.device)
        text_features = self.openclip_model.encode_text(tokenizer_outputs)
        text_features = torch.mean(text_features, dim=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(self):
        pass

class BLIP2_zeroshot(torch.nn.Module):
    def __init__(self, num_frames=1, input_res=224, model_name="openai/clip-vit-base-patch32", device='cuda'):
        super(BLIP2_zeroshot, self).__init__()
        self.num_frames = num_frames
        self.input_res = input_res
        self.device = device
        self.model_name = "weights/blip2-opt-2.7b/"
        self.blip = Blip2Model.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.blip_image_processor = Blip2Processor.from_pretrained(self.model_name)

    def get_visual_features(self, visual_input, is_custom_transforms=False):
        processed_visual_input = self.blip_image_processor(visual_input, size=self.input_res)
        pixel_values = torch.tensor(np.array(processed_visual_input['pixel_values'])).to(self.device)
        pixel_values = pixel_values.reshape(-1, 3, self.input_res, self.input_res)
        visual_features = self.blip.get_image_features(pixel_values)
        visual_features = visual_features.pooler_output.reshape(1, self.num_frames, -1)
        visual_features = torch.mean(visual_features, 1)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
        return visual_features

    def get_text_features(self, text_input):
        tokenizer_outputs = self.tokenizer(text_input, truncation=True, padding=True, return_tensors="pt")
        tokenizer_outputs = {k: torch.LongTensor(np.array(v)).to(self.device) for k, v in tokenizer_outputs.items()}
        text_features = self.blip.get_text_features(**tokenizer_outputs).text_outputs
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(self):
        pass