import pickle as pkl
import numpy as np
from PIL import Image
from util import get_model_stats

def _convert_to_rgb(image):
    return image.convert("RGB")

def image_grid(imgs, rows, cols, name):
    assert len(imgs) == rows * cols

    img_name = "grid_image.png"

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    
    grid.save(img_name)

    return img_name

def get_all_imagenet_attentions(modelname):
    total_layers = get_model_stats(modelname)

    with open('attention_weights/' + modelname + '-layer' + str(total_layers) + '.pkl', 'rb') as f:
        attentions_last = pkl.load(f)

    with open('attention_weights/' + modelname + '-layer' + str(total_layers - 1) + '.pkl', 'rb') as f:
        attentions_second_to_last = pkl.load(f)
    
    with open('attention_weights/' + modelname + '-layer' + str(total_layers - 2) + '.pkl', 'rb') as f:
        attentions_third_to_last = pkl.load(f)
    
    with open('attention_weights/' + modelname + '-layer' + str(total_layers - 3) + '.pkl', 'rb') as f:
        attentions_fourth_to_last = pkl.load(f)
    
    return np.stack([attentions_last, attentions_second_to_last, attentions_third_to_last, attentions_fourth_to_last], axis=1)
