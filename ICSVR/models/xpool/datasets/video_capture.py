import random
import decord
import random
import torch
import numpy as np
import math
decord.bridge.set_bridge("torch")
import cv2
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path,
                               num_frames,
                               sample='rand'):
        """
            video_path: str/os.path
            num_frames: int - number of frames to sample
            sample: 'rand' | 'uniform' how to sample
            returns: frames: torch.tensor of stacked sampled video frames 
                             of dim (num_frames, C, H, W)
                     idxs: list(int) indices of where the frames where sampled
        """
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get indexes of sampled frames
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        # ranges constructs equal spaced intervals (start, end)
        # we can either choose a random image in the interval with 'rand'
        # or choose the middle frame with 'uniform'
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:  # sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                #cv2.imwrite(f'images/{index}.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                raise ValueError

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
            
        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs

def get_frame_indices(num_frames, vlen, sample='rand'):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []

    # ranges constructs equal spaced intervals (start, end)
    # we can either choose a random image in the interval with 'rand'
    # or choose the middle frame with 'uniform'
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    else:  # sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    
    if len(frame_idxs) < num_frames:  # padded with last frame
        padded_frame_indices = [frame_idxs[-1]] * num_frames
        padded_frame_indices[:len(frame_idxs)] = frame_idxs
        frame_idxs = padded_frame_indices
    return frame_idxs

def read_frames_decord(video_path, num_frames, sample='rand'):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    frames = frames.float() / 255
    return frames, frame_indices

def read_image(image_path):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    image = torch.from_numpy(image)
    image = image.float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    #image = torch.stack(image)
    #print(image.shape)
    return image