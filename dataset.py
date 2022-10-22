#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from data.transform import data_transform
import six
import sys
from PIL import Image
import numpy as np
import glob
import os.path as osp
import cv2
import config as cfg
import os


def read_image(img_path, part_size=0):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if img_path == 'no':
        h = part_size[1]
        w = part_size[0]
        img = np.zeros((h, w, 3), np.uint8)
        return Image.fromarray(img, mode='RGB')
    else:
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                    img_path))
                pass
        return img


class TextImageSet(Dataset):
    def __init__(self, root, transform=None, target_transform=None, is_train=False):
        super(TextImageSet, self).__init__()
        self.image_paths = glob.glob(root + '/*.jp*')

        random.shuffle(self.image_paths)

        self.transform = transform
        self.label_transform = target_transform
        self.is_train = is_train

        # label 转换映射
        self.chars2idx = cfg.chars2idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        if self.is_train:
            label = []
            img_p = self.image_paths[index]

            text_img = read_image(img_p)

            plate = os.path.basename(img_p).split('_')[0].lower()

            [label.append(self.chars2idx[p]) for p in plate]

            if self.transform is not None:
                text_img = self.transform(text_img)

            return text_img, label

        else:
            label = []
            img_p = self.image_paths[index]

            text_img = read_image(img_p)

            plate = os.path.basename(img_p).split('_')[0]

            [label.append(self.chars2idx[p]) for p in plate]

            if self.transform is not None:
                text_img = self.transform(text_img)

            return text_img, label, img_p


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __call__(self, batch):
        images, labels = zip(*batch)

        # transform = resizeNormalize((imgW, imgH))
        # transform = data_transform(True)
        # images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
