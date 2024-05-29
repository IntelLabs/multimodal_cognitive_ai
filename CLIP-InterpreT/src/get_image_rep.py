import numpy as np
import torch
import os
import argparse
from torch.utils.data import DataLoader
import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.factory import create_model_and_transforms
from utils.visualization import visualization_preprocess
from prs_hook import hook_prs_logger
import pickle as pkl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(os.path.isdir('attention_weights') == False):
    os.mkdir('attention_weights')

def main():
    parser = argparse.ArgumentParser(description='CLIP model representations')
    parser.add_argument('--modelname', type=str, default='ViT-B-16_openai', help='CLIP model')
    parser.add_argument('--imagepath', type=str, default='data/', help='Image path')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    args = parser.parse_args()

    joint_name = args.modelname # Name of the CLIP model
    batch_size = args.batchsize # Batch size
    image_path = args.imagepath # Path to image folder

    model_name = joint_name.split('_')[0]
    pretrained = "_".join(joint_name.split('_')[1:])

    print(f'Model name: {model_name}')
    print(f'Dataset used: {pretrained}')

    model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()

    ds_vis = ImageFolder(image_path, transform=visualization_preprocess)
    ds = ImageFolder(image_path, transform=preprocess)
    dataloader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=8
    )

    prs = hook_prs_logger(model, device)

    attentions_maps_layer_last = []
    attentions_maps_layer_second_to_last = []
    attentions_maps_layer22_third_to_last = []
    attentions_maps_layer23_fourth_to_last = []
    for index, (images, _) in tqdm.tqdm(enumerate(dataloader)):
        images = images.to(device)
        with torch.no_grad():
            prs.reinit()
            current_representation = model.encode_image(images, 
                                                        attn_method='head', 
                                                        normalize=False)
            current_attentions, _ = prs.finalize(current_representation)

            scores = current_attentions.sum(axis=2)
            
            for ii in range(scores.size()[0]):
                attentions_maps_layer_last.append(scores[ii, -1, :, :].detach().cpu().numpy())
                attentions_maps_layer_second_to_last.append(scores[ii, -2, :, :].detach().cpu().numpy())
                attentions_maps_layer22_third_to_last.append(scores[ii, -3, :, :].detach().cpu().numpy())
                attentions_maps_layer23_fourth_to_last.append(scores[ii, -4, :, :].detach().cpu().numpy())

    total_layers = scores.size()[1]

    attentions_maps_layer_last = np.asarray(attentions_maps_layer_last)
    attentions_maps_layer_second_to_last = np.asarray(attentions_maps_layer_second_to_last)
    attentions_maps_layer22_third_to_last = np.asarray(attentions_maps_layer22_third_to_last)
    attentions_maps_layer23_fourth_to_last = np.asarray(attentions_maps_layer23_fourth_to_last)

    with open('attention_weights/' + model_name + '_' + pretrained + '-layer' + str(total_layers - 1) + '.pkl', 'wb') as f:
        pkl.dump(attentions_maps_layer_last, f)

    with open('attention_weights/' + model_name + '_' + pretrained + '-layer' + str(total_layers - 2) + '.pkl', 'wb') as f:
        pkl.dump(attentions_maps_layer_second_to_last, f)

    with open('attention_weights/' + model_name + '_' + pretrained + '-layer' + str(total_layers - 3) + '.pkl', 'wb') as f:
        pkl.dump(attentions_maps_layer22_third_to_last, f)

    with open('attention_weights/' + model_name + '_' + pretrained + '-layer' + str(total_layers - 4) + '.pkl', 'wb') as f:
        pkl.dump(attentions_maps_layer23_fourth_to_last, f)

if __name__ == '__main__':
    main()
