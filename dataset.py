import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn

import file_utils
import imgproc


class CRAFTData(Dataset):
    def __init__(self, args):
        filelist, _, _ = file_utils.list_files('./data/train/data')
        self.images = []
        self.confmaps = []
        self.scores_region = []
        self.scores_link = []
        for filename in filelist:
            # get datapath
            dataset = os.path.dirname(filename).split(os.sep)[-1]
            filenum = os.path.splitext(os.path.basename(filename))
            label_dir = './data/train/ground_truth/{}/gt_{}/'.format(dataset, filenum)

            # If not exists, generate ground truth
            if not os.path.exists(label_dir):
                continue

            image = imgproc.loadImage(filename)
            score_region = torch.load(label_dir + 'region.pt')
            score_link = torch.load(label_dir + 'link.pt')
            conf_map = torch.load(label_dir + 'conf.pt')

            # resize
            img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                                  interpolation=cv2.INTER_LINEAR,
                                                                                  mag_ratio=args.mag_ratio)

            # Image Preprocess
            x = imgproc.normalizeMeanVariance(img_resized)
            x = x.transpose((2, 0, 1))  # [h, w, c] to [c, h, w]

            h, w, _ = img_resized.shape

            # GT reshape
            score_region = cv2.resize(score_region, dsize=(h / 2, w / 2))
            score_link = cv2.resize(score_link, dsize=(h / 2, w / 2))
            conf_map = cv2.resize(conf_map, dsize=(h / 2, w / 2))

            self.scores_region.append(score_region)
            self.scores_link.append(score_link)
            self.confmaps.append(conf_map)
            self.images.append(x)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx], requires_grad=True)
        y_region = torch.tensor(self.scores_region[idx], requires_grad=False)
        y_link = torch.tensor(self.scores_link[idx], requires_grad=False)
        y_conf = torch.tensor(self.confmaps[idx], requires_grad=False)

        return x, y_region, y_link, y_conf

    def __len__(self):
        return len(self.images)
