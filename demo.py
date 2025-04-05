import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image, ImageOps


import torchvision.transforms as T


from typing import List

import torch.nn as nn
import torchvision
from torch import Tensor
from torchvision.utils import flow_to_image

from smurf import raft_smurf



DEVICE = 'cuda'

def load_image(imfile, transform):
    image = Image.open(imfile)
    image = transform(image)
    img = np.array(image).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    # transform = T.Resize((240, 427))

    model: nn.Module = raft_smurf(checkpoint=args.model)

    model = model.module
    model.to(DEVICE)
    model.eval()

    # with torch.no_grad():
    paths = []
    for entry in os.scandir(args.path):
        if entry.is_dir():
            new_path = args.path + '/' + entry.name
            paths.append(new_path)
    if len(paths) == 0:
        paths.append(args.path)
    print(paths)

    # cwd = os.getcwd()
    # args.output_path = os.path.join(cwd, args.output_path)
    epes = []

    for path in paths:
        _id = 0
        images = glob.glob(os.path.join(path, '*.png')) + \
                    glob.glob(os.path.join(path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = torchvision.io.read_image(imfile1, mode=torchvision.io.ImageReadMode.RGB)
            image2 = torchvision.io.read_image(imfile2, mode=torchvision.io.ImageReadMode.RGB)

            # print(torch.max(image1), torch.min(image1))
            # print(image1.shape)

            flow = model(image1[None], image2[None])
            
            
            epe = torch.sum((flow_pred - flow_gt.cuda())**2, dim=0).sqrt().view(-1).detach().cpu().numpy()
            epes.append(epe)

            # viz(args, image1.detach(), (image1.data - ori).detach(), (flow_pr - flow_up).detach(), flow_pr.detach(), folder_name, str(_id))
            _id += 1
    epes = np.concatenate(epes)
    aepe = np.mean(epes)
    print("##########################")
    print("Average EPE:", aepe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_path', help="output viz")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--raft', help="checkpoint from the RAFT paper?", type=bool, default=True)
    args = parser.parse_args()

    demo(args)