import sys
import os
import time
import argparse
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch import Tensor

from PIL import Image

import cv2
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import test

from craft import CRAFT

from dataset import CRAFTData
from torch.utils.data import DataLoader

from collections import OrderedDict


def train_net(net, image, box_gt, labels, args, refine_net=None):
    gt_region, gt_link, conf_map = generate_gt(net, image, box_gt, labels, args)

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.as_tensor(x).requires_grad_(False)
    x = x.permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
    x = x.requires_grad_(True)
    if args.cuda:
        x = x.cuda()

    # forward pass
    y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0]
    score_link = y[0, :, :, 1]

    # resize ground truth
    gt_region = cv2.resize(gt_region, dsize=score_text.shape,  interpolation=cv2.INTER_LINEAR)
    gt_link = cv2.resize(gt_link, dsize=score_text.shape, interpolation=cv2.INTER_LINEAR)
    conf_map = cv2.resize(conf_map, dsize=score_text.shape, interpolation=cv2.INTER_LINEAR)

    gt_region = torch.tensor(gt_region, requires_grad=False)
    gt_link = torch.tensor(gt_link, requires_grad=False)
    conf_map = torch.tensor(conf_map, requires_grad=False)

    L = torch.sum(conf_map * (torch.pow((score_text - gt_region), 2) + torch.pow((score_link - gt_link), 2)))

    return L


def train(args):
    # load net
    net = CRAFT()  # initialize

    if not os.path.exists(args.trained_model):
        args.trained_model = None

    if args.trained_model is not None:
        print('Loading weights from checkpoint (' + args.trained_model + ')')
        if args.cuda:
            net.load_state_dict(test.copyStateDict(torch.load(args.trained_model)))
        else:
            net.load_state_dict(test.copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    # # LinkRefiner
    # refine_net = None
    # if args.refine:
    #     from refinenet import RefineNet
    #
    #     refine_net = RefineNet()
    #     print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    #     if args.cuda:
    #         refine_net.load_state_dict(test.copyStateDict(torch.load(args.refiner_model)))
    #         refine_net = refine_net.cuda()
    #         refine_net = torch.nn.DataParallel(refine_net)
    #     else:
    #         refine_net.load_state_dict(test.copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
    #
    #     args.poly = True

    optimizer = optim.Adam(net.parameters(), args.learning_rate)
    train_data = CRAFTData(args)
    dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.max_epoch):
        running_loss = 0.0
        for i, data in enumerate(dataloader):

            x, y_region, y_link, y_conf = data

            optimizer.zero_grad()

            y, feature = net(x)

            score_text = y[:, :, :, 0]
            score_link = y[:, :, :, 1]

            L = torch.sum(y_conf * (torch.pow((score_text - y_region), 2) + torch.pow((score_link - y_link), 2)))

            L.backward()
            optimizer.step()

            running_loss += L.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('training finished')





