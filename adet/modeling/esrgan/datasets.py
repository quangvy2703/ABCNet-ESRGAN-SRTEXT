import glob
import random
import os
import numpy as np
import math

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
hr_shape = (480, 480)
hr_height, hr_width = hr_shape
lr_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


hr_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

to_PIL_trans = transforms.ToPILImage()
to_tensor_trans = transforms.ToTensor()


def crop_into_boxes(img):
    _img = to_PIL_trans(img)
    shards = []
    hr_w, hr_h = hr_shape
    w, h = _img.size
    # shards = []
    for h_idx in range(0, h, hr_h):
        # ver_shards = []
        for w_idx in range(0, w, hr_w):
            if w_idx + hr_w > w:
                w_idx = w - hr_w
            if h_idx + hr_h > h:
                h_idx = h - hr_h
            box = (w_idx, h_idx, w_idx + hr_w, h_idx + hr_h)
            cropped_box = _img.crop(box)
            shards.append(to_tensor_trans(cropped_box))
            # ver_shards.append(cropped_box)
        # shards.append(ver_shards)
    return shards


def merge_into_image(cropped_boxes, original_size):
    new_im = Image.new('RGB', original_size)

    x_offset = 0
    y_offset = 0
    for im in cropped_boxes:
        if x_offset + im.size[0] >= original_size[0]:
            x_offset = original_size[0] - im.size[0]
            new_im.paste(im, (x_offset, y_offset))
            y_offset += im.size[1]
            x_offset = 0
            continue
        if y_offset + im.size[1] > original_size[1]:
            y_offset = original_size[1] - im.size[1]
            # new_im.paste(im, (x_offset, y_offset))
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]

def merge_into_image(cropped_boxes, original_size):
    new_im = Image.new('RGB', original_size)

    x_offset = 0
    y_offset = 0
    for im in cropped_boxes:
        if x_offset + im.size[0] >= original_size[0]:
            x_offset = original_size[0] - im.size[0]
            new_im.paste(im, (x_offset, y_offset))
            y_offset += im.size[1]
            x_offset = 0
            continue
        if y_offset + im.size[1] > original_size[1]:
            y_offset = original_size[1] - im.size[1]
            # new_im.paste(im, (x_offset, y_offset))
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]

    return to_tensor_trans(new_im)

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
