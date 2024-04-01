import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
import imageio.v3 as iio

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def read_image(image_path):
    image_np = np.asarray(Image.open(image_path).convert('RGB'))
    image = torch.tensor(image_np)
    image = image.float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image
