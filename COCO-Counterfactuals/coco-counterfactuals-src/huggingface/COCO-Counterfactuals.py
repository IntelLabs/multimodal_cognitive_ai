# coding=utf-8
# Copyright 2023 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import datasets
import json

_CITATION = """\
@inproceedings{le2023cococounterfactuals,
  author = {Tiep Le and Vasudev Lal and Phillip Howard},
  title = {{COCO}-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year = 2023,
  url={https://openreview.net/forum?id=7AjdHnjIHX},
}
"""

_URL = "https://huggingface.co/datasets/Intel/COCO-Counterfactuals"

_DESCRIPTION = """\
COCO-Counterfactuals is a high quality synthetic dataset for multimodal vision-language model evaluation and for training data augmentation. Each COCO-Counterfactuals example includes a pair of image-text pairs; one is a counterfactual variation of the other. The two captions are identical to each other except a noun subject. The two corresponding synthetic images differ only in terms of the altered subject in the two captions. In our accompanying paper, we showed that the COCO-Counterfactuals dataset is challenging for existing pre-trained multimodal models and significantly increase the difficulty of the zero-shot image-text retrieval and image-text matching tasks. Our experiments also demonstrate that augmenting training data with COCO-Counterfactuals improves OOD generalization on multiple downstream tasks.    
"""

class COCO_CounterfactualsConfig(datasets.BuilderConfig):
    """BuilderConfig for COCO-Counterfactuals."""

    def __init__(self, **kwargs):
        """BuilderConfig for COCO-Counterfactuals.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(COCO_CounterfactualsConfig, self).__init__(**kwargs)

class COCO_Counterfactuals(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = COCO_CounterfactualsConfig

    BUILDER_CONFIGS = [
        COCO_CounterfactualsConfig(
            name="default",
        ),
    ]

    IMAGE_EXTENSION = ".jpg"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "image_0": datasets.Image(),
                    "image_1": datasets.Image(),
                    "caption_0": datasets.Value("string"),
                    "caption_1": datasets.Value("string"),
                }
            ),
            homepage=_URL,
            citation=_CITATION,
            task_templates=[],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        hf_auth_token = dl_manager.download_config.use_auth_token
        if hf_auth_token is None:
            raise ConnectionError(
                "Please set use_auth_token=True or use_auth_token='<TOKEN>' to download this dataset"
            )

        downloaded_files = dl_manager.download_and_extract({
            "examples_jsonl": "data/examples.jsonl",
            "images_dir": "data/images.zip",
        })

        return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=downloaded_files)]

    def _generate_examples(self, examples_jsonl, images_dir, no_labels=False):
        """Yields examples."""
        examples = [json.loads(example_json) for example_json in open(examples_jsonl).readlines()]
        for example in examples:
            example["image_0"] = os.path.join(images_dir, "images", example["image_0"] + self.IMAGE_EXTENSION)
            example["image_1"] = os.path.join(images_dir, "images", example["image_1"] + self.IMAGE_EXTENSION)
            id_ = example["id"]
            yield id_, example