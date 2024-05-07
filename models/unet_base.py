#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/4/29 20:41
@Message:
Based on https://github.com/McGregorWwww/UDTransNet
"""
import torch
from torch import nn

from models.awesome import JUp_Concat


class DoubleConvBatchReLu(nn.Module):
    """
    a Block with twice Convolution, BatchNorm and ReLu
    """

    def __init__(self, in_channel, out_channel):
        super(DoubleConvBatchReLu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
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


class UnetDown(nn.Module):
    """
    Unet Down Block
    """

    def __init__(self, in_channel, out_channel):
        super(UnetDown, self).__init__()
        self.conv_block = DoubleConvBatchReLu(in_channel, out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x


class UnetBase(nn.Module):
    def __init__(self, num_channel=3, num_classes=9):
        super(UnetBase, self).__init__()
        self.num_classes = num_classes
        channel = 64
        # 64 128 256 512 1024
        filters = [channel, channel * 2, channel * 4, channel * 8, channel * 16]

        self.Conv = DoubleConvBatchReLu(num_channel, filters[0])

        self.Down1 = UnetDown(filters[0], filters[1])
        self.Down2 = UnetDown(filters[1], filters[2])
        self.Down3 = UnetDown(filters[2], filters[3])
        self.Down4 = UnetDown(filters[3], filters[4])

        self.Up4 = UnetUp(filters[4], filters[3])
        self.Up3 = UnetUp(filters[3], filters[2])
        self.Up2 = UnetUp(filters[2], filters[1])
        self.Up1 = UnetUp(filters[1], filters[0])

        self.pred = nn.Sequential(
            nn.Conv2d(filters[0], filters[0] // 2, kernel_size=1),
            nn.BatchNorm2d(filters[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0] // 2, num_classes, kernel_size=1),
        )

    def forward(self, x):
        e1 = self.Conv(x)
        # encode
        e2 = self.Down1(e1)
        e3 = self.Down2(e2)
        e4 = self.Down3(e3)
        e5 = self.Down4(e4)
        # decode
        d4 = self.Up4(e5, e4)
        d3 = self.Up3(d4, e3)
        d2 = self.Up2(d3, e2)
        d1 = self.Up1(d2, e1)

        if self.num_classes == 1:
            out = nn.Sigmoid()(self.pred(d1))
        else:
            out = self.pred(d1)

        return out

class UnetEncoder(nn.Module):
    def __init__(self, num_channel=3):
        super(UnetEncoder, self).__init__()
        channel = 64
        # 64 128 256 512 1024
        filters = [channel, channel * 2, channel * 4, channel * 8, channel * 16]

        self.Conv = DoubleConvBatchReLu(num_channel, filters[0])

        self.Down1 = UnetDown(filters[0], filters[1])
        self.Down2 = UnetDown(filters[1], filters[2])
        self.Down3 = UnetDown(filters[2], filters[3])
        self.Down4 = UnetDown(filters[3], filters[4])


    def forward(self, x):
        e1 = self.Conv(x)
        # encode
        e2 = self.Down1(e1)
        e3 = self.Down2(e2)
        e4 = self.Down3(e3)
        e5 = self.Down4(e4)

        return [e1, e2, e3, e4, e5]

if __name__ == "__main__":
    print("unet_base.py")

    model = UnetBase(num_channel=3, num_classes=9)
    # print(model)

    input_data = torch.randn((1, 3, 256, 256))
    output_data = model(input_data)
    print(output_data.shape)
    print(f'The parameters size of model is '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')
