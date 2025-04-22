import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import core.datasets as datasets
from utils import flow_viz
from utils import frame_utils

from smurf import raft_smurf

from utils.utils import InputPadder, forward_interpolate
from demo import *

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow = model(image1, image2)
            flow = padder.unpad(flow_pr[-1][0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flow = model(image1, image2)
        flow = padder.unpad(flow[-1][0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_nuscenes_submission(model, iters=24, partition='front', output_path='nuscenes_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.nuScenes(aug_params=None, partition=partition)
    output_path = os.path.join(output_path, partition)

    print(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flow = model(image1, image2)
        flow = padder.unpad(flow[-1][0]).permute(1, 2, 0).cpu().numpy()
        # flo = flow_viz.flow_to_image(flow)
        # flox_rgb = Image.fromarray(flo.astype('uint8'), 'RGB')
        # flox_rgb.save(output_path + '/' + frame_id)
        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlow(output_filename, flow)
        # saved = frame_utils.readFlow(output_filename)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        flow = model(image1, image2)
        epe = torch.sum((flow[-1][0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32, train=True):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    if args.attack_type != 'None':
        torch.set_grad_enabled(True)

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, train=train)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = model(image1, image2)
            flow = flow[-1]
            flow = padder.unpad(flow[0]).cpu()


            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).detach().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    if args.attack_type != 'None':
        torch.set_grad_enabled(True) 
    val_dataset = datasets.KITTI(split='training')


    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        # flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = model(image1, image2)
        flow = flow[-1]
        flow = padder.unpad(flow[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        # break

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--raft', help="checkpoint from the RAFT paper?", type=bool, default=True)
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--attack_type', help='Attack type options: None, FGSM, PGD', type=str, default='PGD')
    parser.add_argument('--iters', help='Number of iters for PGD?', type=int, default=50)
    parser.add_argument('--epsilon', help='epsilon?', type=int, default=10.0)
    parser.add_argument('--channel', help='Color channel options: 0, 1, 2, -1 (all)', type=int, default=-1)    
    parser.add_argument('--fcbam', help='Add CBAM after the feature network?', type=bool, default=False)
    parser.add_argument('--ccbam', help='Add CBAM after the context network?', type=bool, default=False)
    parser.add_argument('--deform', help='Add deformable convolution?', type=bool, default=False)
    parser.add_argument('--output_path', help="output viz")
    parser.add_argument('--name', help="output viz", default="flow.png")
    parser.add_argument('--partition', type=str, default="front")
    parser.add_argument('--checkpoint', type=str, default="smurf_kitti.pt")



    args = parser.parse_args()

    # model = torch.nn.DataParallel(RAFT(args))

    # if not args.raft:
    #     checkpoint = torch.load(args.model)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     model.load_state_dict(torch.load(args.model))

    model = raft_smurf(checkpoint=args.checkpoint)


    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, train=False)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == "nuscenes":
            create_nuscenes_submission(model.module, partition=args.partition)


