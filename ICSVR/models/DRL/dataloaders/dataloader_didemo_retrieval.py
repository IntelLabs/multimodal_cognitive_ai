from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
import tempfile
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset

class DiDeMoDataset(RetrievalDataset):
    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(DiDeMoDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                            max_frames, video_framerate, image_resolution, mode, config=config)
        pass

    
    def _get_anns(self, subset='train'):
        """
        video_dict: dict: video_id -> video_path
        sentences_dict: list: [(video_id, caption)] , caption (list: [text:, start, end])
        """

        train_video_ids = []
        with open(join(self.anno_path, 'didemo_train_list.txt'), 'r') as f:
            for line in f.readlines():
                train_video_ids.append(line[:-1])

        val_video_ids = []
        with open(join(self.anno_path, 'didemo_val_list.txt'), 'r') as f:
            for line in f.readlines():
                val_video_ids.append(line[:-1])

        test_video_ids = []
        with open(join(self.anno_path, 'didemo_test_list.txt'), 'r') as f:
            for line in f.readlines():
                test_video_ids.append(line[:-1])

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        if subset == 'train':
            anno_path = join(self.anno_path, 'DIDEMO_drl_train.json')
            data = json.load(open(anno_path, 'r'))
            for video_id, video_id_value in data.items():
                if video_id in train_video_ids:
                    caption = video_id_value['text'][0]
                    sentences_dict[len(sentences_dict)] = (video_id, (caption, None, None))
                    video_dict[video_id] = join(self.video_path, "{}.mp4".format(video_id))
        elif subset == 'val':
            anno_path = join(self.anno_path, 'DIDEMO_drl_val.json')
            data = json.load(open(anno_path, 'r'))
            for video_id, video_id_value in data.items():
                if video_id in val_video_ids:
                    caption = video_id_value['text'][0]
                    sentences_dict[len(sentences_dict)] = (video_id, (caption, None, None))
                    video_dict[video_id] = join(self.video_path, "{}.mp4".format(video_id))
        else:
            print("#################################################################################")
            print("Using type: {} ".format(self.config.typ))
            print("#################################################################################")
            anno_path = join(self.anno_path, "DIDEMO_clip4clip_" + self.config.typ + ".json")
            data = json.load(open(anno_path, 'r'))
            for video_id, video_id_value in data.items():
                if video_id in test_video_ids:
                    caption = video_id_value['text'][0]
                    sentences_dict[len(sentences_dict)] = (video_id, (caption, None, None))
                    video_dict[video_id] = join(self.video_path, "{}.mp4".format(video_id))

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print('[{}] Unique sentence is {} , all num is {}'.format(subset, len(unique_sentence), len(sentences_dict)))

        return video_dict, sentences_dict