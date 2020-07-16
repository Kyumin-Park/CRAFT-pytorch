import sys
import os
import time
import argparse
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
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
from tqdm import tqdm

from craft import CRAFT

from dataset import CRAFTDataset
from torch.utils.data import DataLoader

from collections import OrderedDict


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

    criterion = craft_utils.CRAFTLoss()
    optimizer = optim.Adam(net.parameters(), args.learning_rate)
    train_data = CRAFTDataset(args)
    dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for epoch in range(args.max_epoch):
        running_loss = 0.0
        for i, data in pbar:
            x, y_region, y_link, y_conf = data
            x = x.cuda()
            y_region = y_region.cuda()
            y_link = y_link.cuda()
            y_conf = y_conf.cuda()
            # x, label_dir = data
            optimizer.zero_grad()

            y, feature = net(x)

            score_text = y[:, :, :, 0]
            score_link = y[:, :, :, 1]

            L = criterion(score_text, score_link, y_region, y_link, y_conf)
            # L = criterion(score_text, score_link, label_dir)

            L.backward()
            optimizer.step()

            running_loss += L.data.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('training finished')





