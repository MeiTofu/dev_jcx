#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Jeff
@Description:
Based on https://github.com/McGregorWwww/UDTransNet
"""
import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        # print(x,y)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class ImageToImage2D_kfold(Dataset):
    def __init__(self,
                 dataset_path: str,
                 split_txt: list,
                 joint_transform: Callable = None,
                 image_size: int = 512,
                 split: str = 'train',
                 images_path="images",
                 masks_path="masks",
                 task_name: str = 'Synapse') -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.split_txt = split_txt

        self.input_path = os.path.join(dataset_path, images_path)
        self.output_path = os.path.join(dataset_path, masks_path)
        # self.images_list = os.listdir(self.input_path)
        self.task_name = task_name
        self.split = split

        if joint_transform:
            self.joint_transform = joint_transform
        # else:
        #     to_tensor = T.ToTensor()
        #     self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.split_txt)

    def __getitem__(self, idx):
        image_filename = self.split_txt[idx].split()[0]
        file_prefix, file_suffix  =os.path.splitext(image_filename)
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        mask = cv2.imread(os.path.join(self.output_path, file_prefix + ".png"), 0)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        # if self.task_name == "Synapse":
        #     if self.split != 'train':
        #         self.joint_transform = None
        # else:
        #     image = cv2.resize(image, (self.image_size,self.image_size))
        #     mask = cv2.resize(mask, (self.image_size,self.image_size))
        #     mask[mask <= 0] = 0
        #     mask[mask > 0] = 1
        #     image, mask = correct_dims(image, mask)

        sample = {'image': image, 'label': mask}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename


if __name__ == "__main__":
    print("Load_Dataset.py")
    train_path = "../../dataset/Synapse_dev"
    val_path = "../../dataset/Synapse_all"
    val_txt_path = "test_dev.txt"
    train_txt_path = "train_dev.txt"
    # 读取数据集对应的txt
    with open(os.path.join(val_path, val_txt_path), "r") as f:
        val_lines = f.readlines()
    with open(os.path.join(val_path, train_txt_path), "r") as f:
        train_lines = f.readlines()

    val_tf = ValGenerator(output_size=[512, 512])
    train_tf = RandomGenerator(output_size=[512, 512])
    train_dataset = ImageToImage2D_kfold(dataset_path=train_path, split_txt=train_lines, joint_transform=train_tf, split='train')
    val_dataset = ImageToImage2D_kfold(dataset_path=val_path, split_txt=val_lines, joint_transform=val_tf, split='val')
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)

    for i, (sampled_batch, names) in enumerate(val_loader, 1):
        images, masks = sampled_batch['image'], sampled_batch['label']
        # images, masks = images.cuda(), masks.cuda()
        temp_images = images[0].cpu().numpy()
        temp_mask = masks[0].cpu().numpy()
        temp_masks = masks.cpu().numpy()
        print(i)

    print()
