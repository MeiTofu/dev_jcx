#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 19:39
@Message: null
"""
import os

import numpy as np

# CLASSES = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

CLASSES = ["_background_", "H", "T1", "T2", "T3", "T4", "T5"]


def cvtColor(image):
    """
    将图像转换成RGB图像，防止灰度图在预测时报错。
    代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    :param image: np 类型的图像数据
    :return: 转换为RGB格式的 np 类型的图像数据
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def preprocess_input(image):
    """
    TODO 数据标准化？
    :param image:
    :return:
    """
    image /= 255.0
    return image


def generate_dir(dir_path):
    """
    若目标文件夹不存在，则自动创建
    :param dir_path: 需要创建的文件夹路径
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
