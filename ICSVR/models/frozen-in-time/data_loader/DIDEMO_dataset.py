import json
import os
import random

import numpy as np
import pandas as pd

from base.base_dataset import TextVideoDataset


class DIDEMO(TextVideoDataset):
    def _load_metadata(self):
        if(self.split == 'train'):
            json_fp = os.path.join(self.data_dir, 'didemo_fit_train.json')
        elif(self.split == 'test'):
            json_fp = os.path.join(self.data_dir, 'DIDEMO_fit_' + self.typ + '.json')
        else:
            json_fp = os.path.join(self.data_dir, 'didemo_fit_val.json')

        print("Split: {} Loading file: {} ".format(self.split, json_fp))
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        train_list_path = "didemo_train_list.txt"
        val_list_path = "didemo_val_list.txt"
        test_list_path = "didemo_test_list.txt"

        train_df = pd.read_csv(os.path.join(self.data_dir, train_list_path), names=['videoid'])
        val_df = pd.read_csv(os.path.join(self.data_dir, val_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(self.data_dir, test_list_path), names=['videoid'])
        self.split_sizes = {'train': len(train_df), 'val': len(test_df), 'test': len(test_df)}

        if self.split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        elif self.split == 'test':
            df = df[df['image_id'].isin(test_df['videoid'])]
        else:
            df = df[df['image_id'].isin(val_df['videoid'])]

        print("#####################################################################################")
        print("Split type: {} and Data Length: {} ".format(self.split, len(df)))
        self.metadata = df.groupby(['image_id'])['caption'].apply(list)
        if self.subsample < 1:
            self.metadata = self.metadata.sample(frac=self.subsample)

        self.metadata = pd.DataFrame({'captions': self.metadata})

    def _get_video_path(self, sample):
        return os.path.join(self.video_data_dir, sample.name + '.mp4'), sample.name + '.mp4'

    def _get_caption(self, sample):
        caption = sample['captions'][0]
        return caption
