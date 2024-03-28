#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 19:39
@Message: null
"""
import os
import random

import numpy as np

# CLASSES = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

CLASSES = ["_background_", "H", "T1", "T2", "T3", "T4", "T5"]

CLASSES_NAME = ["H", "T1", "T2", "T3", "T4", "T5"]

NAME_CLASSES = ["background", "H", "T1", "T2", "T3", "T4", "T5"]

COLORS_CV = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
             (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
             (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
             (128, 64, 12)]

COLORS_PIL = [(255, 69, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)]


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('You have chosen to seed training. This will slow down your training!')


def cvtColor(image):
    """
    将图像转换成RGB图像，防止灰度图在预测时报错。
    代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    :param image: PIL 类型的图像对象
    :return: 转换为RGB格式的 PIL 类型的图像对象
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image,
                 size: tuple = (512, 512)):
    """
    对输入图像进行不失真 resize
    :param image: PIL 类型的图像对象
    :param size: 目标尺寸
    :return: 调整尺寸后的图片，添加的灰度条的宽高
    """
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def preprocess_input(image):
    """
    数据标准化？
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


def draw_filled_rectangle(draw,
                          rectangle_position: tuple = (50, 50),
                          rectangle_size: int = 100,
                          rectangle_color: tuple = (255, 0, 0)):
    """
        输入 ImageDraw 对象绘制一个填充指定颜色的矩形并返回结果。
    :param draw: ImageDraw.Draw(image)
    :param rectangle_position: 矩形位置
    :param rectangle_size: 矩形大小
    :param rectangle_color: 矩形填充颜色
    :return: ImageDraw
    """
    rectangle_coordinates = [
        rectangle_position,
        (rectangle_position[0] + rectangle_size, rectangle_position[1]),
        (rectangle_position[0] + rectangle_size, rectangle_position[1] + rectangle_size // 2),
        (rectangle_position[0], rectangle_position[1] + rectangle_size // 2),
    ]

    draw.polygon(rectangle_coordinates, outline=None, fill=rectangle_color)
    return draw


def add_info_to_image(image,
                      scale_factor: int = 20,
                      font_color: tuple = (0, 0, 0),
                      colors=None,
                      name_classes=None):
    """
        向图像中添加文本信息并返回处理好的图像
    :param image: Image
    :param scale_factor: 文本相对于图像尺寸计算得到的缩放因子
    :param font_color: 文本颜色，PIL 颜色编码
    :param colors: 矩形填充颜色，默认即可
    :param name_classes: 类别名称，默认即可
    :return: Image
    """
    if colors is None:
        colors = COLORS_PIL
    if name_classes is None:
        name_classes = CLASSES_NAME

    w, h = image.size
    unit = w // scale_factor
    distance = (w - unit * 2) // len(name_classes)
    # 自适应字体大小
    font_size = unit
    # 字体位置：底部
    x, y = (unit, h - font_size)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("utils/arial.ttf", size=font_size)

    for i in range(len(name_classes)):
        text = name_classes[i]
        position = (x, y - unit)
        position_text = (x, y)
        rectangle_size = unit
        draw = draw_filled_rectangle(draw, position, rectangle_size, rectangle_color=colors[i])
        draw.text(position_text, text, font=font, fill=font_color)
        x = x + distance

    return image


def save_info_when_training(train_loss, val_loss, val_acc, lr_list, save_path='', save=True):
    X = range(0, len(train_loss))
    y_train = train_loss
    plt.figure(figsize=(12, 7), dpi=200)
    plt.subplot(1, 3, 1)
    plt.plot(X, lr_list, 'y.-', label='lr')
    plt.title('lr vs epochs item')
    plt.ylabel('lr')

    x_loss = range(0, len(val_loss))
    plt.subplot(1, 3, 2)
    plt.plot(x_loss, y_train, 'b.-', label='train')
    plt.title('train/val loss vs epochs item')
    plt.ylabel('train/val loss')
    y_val = val_loss
    plt.plot(X, y_val, 'r.-', label='val')
    plt.legend()

    x_acc = range(0, len(val_acc))
    plt.subplot(1, 3, 3)
    plt.plot(x_acc, val_acc, 'g.-', label='val')
    plt.title('val f_score vs epochs item')
    plt.ylabel('val f_score')

    plt.tight_layout()
    if save:
        plt.savefig(save_path + '/train_info.png')
    else:
        plt.show()
