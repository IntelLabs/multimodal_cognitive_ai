from base import MultiDistBaseDataLoaderExplicitSplit, BaseDataLoaderExplicitSplit
from data_loader.transforms import init_transform_dict
from data_loader.MSRVTT_dataset import MSRVTT
from data_loader.MSVD_dataset import MSVD_train, MSVD
from data_loader.DIDEMO_dataset import DIDEMO

def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   typ,
                   video_data_dir,
                   question,
                   metadata_dir=None,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1,
                   sliding_window_stride=-1,
                   reader='cv2'):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        typ=typ,
        video_data_dir=video_data_dir,
        question=question,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "MSVD_train":
        dataset = MSVD_train(**kwargs)
    elif dataset_name == "MSVD":
        dataset = MSVD(**kwargs)
    elif dataset_name == "DIDEMO":
        dataset = DIDEMO(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class MultiDistTextVideoDataLoader(MultiDistBaseDataLoaderExplicitSplit):

    def __init__(self,
                 args,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 typ,
                 video_data_dir,
                 question,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, typ, video_data_dir, question,
                                 metadata_dir, split, tsfm, cut, subsample, sliding_window_stride, reader)
        super().__init__(args, dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 typ,
                 video_data_dir,
                 question,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, typ, video_data_dir, 
                                 question, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)

        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name
       
      
class TextVideoDataLoader_CLIP(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 typ,
                 video_data_dir,
                 question,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict_clip(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, typ, video_data_dir,
                                 question, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)

        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name
