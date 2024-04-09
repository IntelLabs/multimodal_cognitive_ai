from asyncore import read
import os
import random
from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from datasets.video_capture import VideoCapture, read_frames_decord

class DIDEMODataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
        Notes: for test split, we return one video, caption pair for each caption belonging to that video
               so when we run test inference for t2v task we simply average on all these pairs.
    """

    def __init__(self, config, split_type = 'train', img_transforms=None):
        self.config = config
        self.path = "../../data/xpool_data/"
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.datapath = self.path
        json_file = self.datapath + 'DIDEMO_xpool_' + config.typ + '.json'
        print("####################################################################################")
        print("Loading file: {} ".format(json_file))
        test_file = self.datapath + 'didemo_test_list.txt'
        val_file = self.datapath + 'didemo_val_list.txt'
        train_file = self.datapath + 'didemo_train_list.txt'
        self.vid2caption = load_json(json_file)
        self.all_train_pairs = []
        self.all_test_pairs = []
        self.all_val_pairs = []
        self.limit = 0

        if split_type == 'train':
            self.train_vids = read_lines(train_file)
            print("Length of training videos: {} ".format(len(self.train_vids)))
            self._construct_all_train_pairs(self.vid2caption)
            print("Total size of training data: {} ".format(len(self.all_train_pairs)))
            print("####################################################################################")
        elif split_type ==  'val':
            self.val_vids = read_lines(val_file)
            print("Length of validation videos: {} ".format(len(self.val_vids)))
            self._construct_all_val_pairs(self.vid2caption)
            print("Total size of validation data: {} ".format(len(self.all_val_pairs)))
            print("####################################################################################")
        elif split_type == 'test':
            self.test_vids = read_lines(test_file)
            print("Length of testing videos: {} ".format(len(self.test_vids)))
            self._construct_all_test_pairs(self.vid2caption)
            print("Total size of test data: {}".format(len(self.all_test_pairs)))
            print("####################################################################################")

    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_train(index)
        elif self.split_type == 'val':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_val(index)
        elif self.split_type == 'test':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_test(index)
        imgs, idxs = read_frames_decord(video_path, 
                                        self.config.num_frames, 
                                        self.config.video_sample_type)
        
        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        ret = {
            'video_id': video_id,
            'video': imgs,
            'text': caption
        }

        return ret

    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, en_caption = self.all_train_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        return video_path, en_caption, vid

    def _get_vidpath_and_caption_by_index_val(self, index):
        vid, en_caption = self.all_val_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        return video_path, en_caption, vid

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, en_caption = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        return video_path, en_caption, vid

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        elif self.split_type == 'val':
            return len(self.all_val_pairs)
        elif self.split_type == 'test':
            return len(self.all_test_pairs)

    def _construct_all_train_pairs(self, en_dict):
        for vid in self.train_vids:
            caption = en_dict[vid]
            self.all_train_pairs.append([vid, caption])

    def _construct_all_val_pairs(self, en_dict):
        for vid in self.val_vids:
            caption = en_dict[vid]
            self.all_val_pairs.append([vid, caption])

    def _construct_all_test_pairs(self, en_dict):
        for vid in self.test_vids:
            caption = en_dict[vid]
            self.all_test_pairs.append([vid, caption])
