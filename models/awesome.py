#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/4/23 21:28
@Message: Awesome module for JUNet
"""
import torch
from torch import nn

from models.basic import SeparableConv2d, SubPixelConvolutionalBlock


class AdjustmentBlock(nn.Module):
    """
    用于调整通道和上采样，这里采用可分离卷积进行上采样后的特征调整，以降低计算开销
    """
    def __init__(self, in_channel, out_channel):
        super(AdjustmentBlock, self).__init__()
        # hide_kernel = 3
        self.adjustment_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(out_channel, out_channel, kernel_size=hide_kernel, padding=hide_kernel//2, bias=False),
            # nn.Conv2d( out_channel, out_channel, kernel_size=hide_kernel, stride=1, padding=hide_kernel//2, groups=out_channel, bias=False),
            # nn.BatchNorm2d(out_channel),
            # nn.ReLU(inplace=True)
            SeparableConv2d(out_channel, out_channel)
        )

    def forward(self, x):
        x = self.adjustment_block(x)
        return x


class UnetUp(nn.Module):
    """
    Unet Up Block
    """

    def __init__(self, in_channel, out_channel):
        super(UnetUp, self).__init__()
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(2 * out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = self.up_block(x)
        x = torch.cat((skip, x), dim=1)
        x = self.conv_block(x)
        return x


class JUp_Concat_v1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(JUp_Concat_v1, self).__init__()
        self.subpixel_convolutional_blocks = SubPixelConvolutionalBlock(n_channels=in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.separableConv2d = SeparableConv2d(in_channels=out_channel*2, out_channels=out_channel)
        self.separableConv2d2 = SeparableConv2d(in_channels=out_channel*2, out_channels=out_channel)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(2 * out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    # x skip 1024 512
    def forward(self, x, skip):
        x = self.subpixel_convolutional_blocks(x)
        outputs = torch.cat([skip, x], dim=1)

        # 残差连接
        residual = outputs
        residual = self.separableConv2d(residual)

        outputs = self.conv_block2(outputs)

        outputs = torch.cat([outputs, residual], dim=1)
        outputs = self.separableConv2d2(outputs)
        return outputs


class JUp_Concat(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(JUp_Concat, self).__init__()
        self.subpixel_convolutional_blocks = SubPixelConvolutionalBlock(n_channels=in_channel)
        hide_channel = out_channel // 2
        self.separableConv2d = SeparableConv2d(in_channels=hide_channel*3, out_channels=out_channel)
        self.separableConv2d2 = SeparableConv2d(in_channels=out_channel*2, out_channels=out_channel)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hide_channel*3, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    # x skip 1024 512
    def forward(self, x, skip):
        x = self.subpixel_convolutional_blocks(x)   # 1024 / 4 = 256
        outputs = torch.cat([skip, x], dim=1)   # 512 + 256 = 768

        # 残差连接
        residual = outputs.clone()
        residual = self.separableConv2d(residual)

        outputs = self.conv_block2(outputs)

        outputs = torch.cat([outputs, residual], dim=1)
        outputs = self.separableConv2d2(outputs)
        return outputs


class JUp_LKA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(JUp_LKA, self).__init__()
        self.subpixel_convolutional_blocks = SubPixelConvolutionalBlock(n_channels=in_channel)
        hide_channel = out_channel // 2
        self.separableConv2d = SeparableConv2d(in_channels=hide_channel*3, out_channels=out_channel)
        self.separableConv2d2 = SeparableConv2d(in_channels=out_channel*2, out_channels=out_channel)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hide_channel*3, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    # x skip 1024 512
    def forward(self, x, skip):
        x = self.subpixel_convolutional_blocks(x)   # 1024 / 4 = 256
        outputs = torch.cat([skip, x], dim=1)   # 512 + 256 = 768

        # 残差连接
        residual = outputs.clone()
        residual = self.separableConv2d(residual)

        outputs = self.conv_block2(outputs)

        outputs = torch.cat([outputs, residual], dim=1)
        outputs = self.separableConv2d2(outputs)
        return outputs


def get_head(head_type, filters):
    if head_type == "JUp_Concat":
        Up4 = JUp_Concat(filters[4], filters[3])
        Up3 = JUp_Concat(filters[3], filters[2])
        Up2 = JUp_Concat(filters[2], filters[1])
        Up1 = JUp_Concat(filters[1], filters[0])
    elif head_type == "UnetUp":
        Up4 = UnetUp(filters[4], filters[3])
        Up3 = UnetUp(filters[3], filters[2])
        Up2 = UnetUp(filters[2], filters[1])
        Up1 = UnetUp(filters[1], filters[0])
    else:
        raise ValueError('Unsupported head type with `{}`, please use UnetUp or JUp_Concat.'.format(head_type))
    return Up4, Up3, Up2, Up1

