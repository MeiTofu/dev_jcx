#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 21:27
@Message: null
"""
import torch
from torch import nn

from models.basic import SubPixelConvolutionalBlock, SeparableConv2d
from models.resnet import resnet50


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):    # (1,1024,32,32) (1,2048,16,16)
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)     # (1,3072, 32,32)
        outputs = self.conv1(outputs)   # (1,512,32,32)
        outputs = self.relu(outputs)    # (1,512,32,32)
        outputs = self.conv2(outputs)   # (1,512,32,32)
        outputs = self.relu(outputs)    # (1,512,32,32)
        return outputs  # (1,512,32,32)


class JUp(nn.Module):
    def __init__(self, in_size, out_size, up_channels=2048):
        super(JUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.subpixel_convolutional_blocks = SubPixelConvolutionalBlock(kernel_size=3, n_channels=up_channels)
        self.relu = nn.ReLU(inplace=True)
        self.separableConv2d = SeparableConv2d(in_channels=in_size, out_channels=out_size)

    def forward(self, inputs1, inputs2):    # (1,1024,32,32) (1,2048,16,16)
        outputs2 = self.subpixel_convolutional_blocks(inputs2)
        outputs = torch.cat([inputs1, outputs2], 1)     # 1536    # (1,3072, 32,32)

        # 残差连接
        # residual = outputs
        # residual = self.separableConv2d(residual)

        outputs = self.conv1(outputs)   # (1,512,32,32)
        outputs = self.relu(outputs)    # (1,512,32,32)
        outputs = self.conv2(outputs)   # (1,512,32,32)
        outputs = self.relu(outputs)    # (1,512,32,32)
        # outputs = outputs + residual
        return outputs  # (1,512,32,32)

class JUp_Concate(nn.Module):
    def __init__(self, in_size, out_size, up_channels=2048, repair_channel=1024):
        super(JUp_Concate, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.subpixel_convolutional_blocks = SubPixelConvolutionalBlock(kernel_size=3, n_channels=up_channels)
        self.relu = nn.ReLU(inplace=True)
        self.separableConv2d = SeparableConv2d(in_channels=in_size, out_channels=out_size)
        self.separableConv2d2 = SeparableConv2d(in_channels=repair_channel, out_channels=out_size)

    def forward(self, inputs1, inputs2):    # (1,1024,32,32) (1,2048,16,16)
        outputs2 = self.subpixel_convolutional_blocks(inputs2)
        outputs = torch.cat([inputs1, outputs2], 1)     # 1536    # (1,3072, 32,32)

        # 残差连接
        residual = outputs
        residual = self.separableConv2d(residual)

        outputs = self.conv1(outputs)   # (1,512,32,32)
        outputs = self.relu(outputs)    # (1,512,32,32)
        outputs = self.conv2(outputs)   # (1,512,32,32)
        outputs = self.relu(outputs)    # (1,512,32,32)
        # outputs = outputs + residual
        outputs = torch.cat([outputs, residual], dim=1)
        outputs = self.separableConv2d2(outputs)
        return outputs  # (1,512,32,32)


class Head(nn.Module):
    def __init__(self,
                 backbone_type="resnet50",
                 num_classes=7):
        super(Head, self).__init__()

        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.out_filters = [64, 128, 256, 512]

        if backbone_type == 'vgg':
            self.in_filters = [192, 384, 768, 1024]
        elif backbone_type == "resnet50":
            self.in_filters = [192, 512, 1024, 3072]

        # upsampling
        # # 64,64,512
        # self.up_concat4 = unetUp(self.in_filters[3], self.out_filters[3])
        # # 128,128,256
        # self.up_concat3 = unetUp(self.in_filters[2], self.out_filters[2])
        # # 256,256,128
        # self.up_concat2 = unetUp(self.in_filters[1], self.out_filters[1])
        # # 512,512,64
        # self.up_concat1 = unetUp(self.in_filters[0], self.out_filters[0])

        # self.up_concat4 = JUp(1536, self.out_filters[3], up_channels=2048)
        # # 128,128,256
        # self.up_concat3 = JUp(640, self.out_filters[2], up_channels=512)
        # # 256,256,128
        # self.up_concat2 = JUp(320, self.out_filters[1], up_channels=256)
        # # 512,512,64
        # self.up_concat1 = JUp(96, self.out_filters[0], up_channels=128)

        self.up_concat4 = JUp_Concate(1536, self.out_filters[3], up_channels=2048, repair_channel=1024)
        # 128,128,256
        self.up_concat3 = JUp_Concate(640, self.out_filters[2], up_channels=512, repair_channel=512)
        # 256,256,128
        self.up_concat2 = JUp_Concate(320, self.out_filters[1], up_channels=256, repair_channel=256)
        # 512,512,64
        self.up_concat1 = JUp_Concate(96, self.out_filters[0], up_channels=128, repair_channel=128)

        if self.backbone_type == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.out_filters[0], self.out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.out_filters[0], self.out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(self.out_filters[0], self.num_classes, 1)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = inputs    # 64 256 512 1024 2048
        up4 = self.up_concat4(feat4, feat5)     # (1,512,32,32)
        up3 = self.up_concat3(feat3, up4)   # (1,256,64,64)
        up2 = self.up_concat2(feat2, up3)   # (1,128,128,128)
        up1 = self.up_concat1(feat1, up2)   # (1,64,256,256)
        # (1,64,256,256)
        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        output_head = self.final(up1)

        return output_head


if __name__ == "__main__":
    print("dev head")

    backbone = resnet50(pretrained=False)
    model = Head(backbone_type='resnet50', num_classes=7)
    # print(model)

    input_data = torch.randn((1, 3, 512, 512))
    output_backbone = backbone(input_data)
    output_data = model(output_backbone)

    print(output_data.shape)

    print(f'The parameters size of model is ' 
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')
