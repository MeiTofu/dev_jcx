#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/28 21:21
@Message: null
"""
import torch
from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                                        stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.siLu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.siLu(x)
        return x


class SubPixelConvolutionalBlock_dep(nn.Module):
    """
    子像素卷积模块, 包含卷积, 像素清洗和激活层.
    """

    def __init__(self, n_channels=64, kernel_size=1, scaling_factor=2):
        """
        :参数 kernel_size: 卷积核大小
        :参数 n_channels: 输入和输出通道数
        :参数 scaling_factor: 放大比例
        """
        super(SubPixelConvolutionalBlock_dep, self).__init__()

        # 首先通过卷积将通道数扩展为 scaling factor^2 倍
        # self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
        #                       kernel_size=kernel_size, padding=kernel_size // 2)
        # 进行像素清洗，合并相关通道数据
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        # 最后添加激活层
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像数据集，张量表示，大小为(N, n_channels, w, h)
        :返回: 输出图像数据集，张量表示，大小为 (N, n_channels, w * scaling factor, h * scaling factor)
        """
        # output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(input)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output

class SubPixelConvolutionalBlock(nn.Module):
    """
    子像素卷积模块, 包含卷积, 像素清洗和激活层.
    """

    def __init__(self, n_channels=64, kernel_size=1, scaling_factor=2):
        """
        :参数 kernel_size: 卷积核大小
        :参数 n_channels: 输入和输出通道数
        :参数 scaling_factor: 放大比例
        """
        super(SubPixelConvolutionalBlock, self).__init__()
        # in_channel = n_channels // (scaling_factor **2)
        # hide_channel = n_channels // scaling_factor
        # # 首先通过卷积将通道数扩展为 scaling factor^2 倍
        # self.conv = nn.Conv2d(in_channels=in_channel, out_channels=hide_channel,
        #                       kernel_size=kernel_size, padding=kernel_size // 2)
        # self.bn = nn.BatchNorm2d(hide_channel)
        # self.relu = nn.ReLU(True)
        # 进行像素清洗，合并相关通道数据
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        # 最后添加激活层
        self.prelu = nn.PReLU()

    def forward(self, x):
        """
        前向传播.

        :参数 input: 输入图像数据集，张量表示，大小为(N, n_channels, w, h)
        :返回: 输出图像数据集，张量表示，大小为 (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.pixel_shuffle(x)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        # output = self.relu(self.bn(self.conv(output)))  # (N, n_channels * scaling factor^2, w, h)
        return output

