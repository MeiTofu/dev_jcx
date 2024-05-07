#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 21:27
@Message: null
"""
import torch
from torch import nn

from models.awesome import AdjustmentBlock, get_head
from models.resnet import resnet50
from models.unet_base import UnetEncoder


class Head(nn.Module):
    def __init__(self,
                 backbone_type="resnet50",
                 num_classes=7,
                 head_up="unetUp"):
        super(Head, self).__init__()

        self.backbone_type = backbone_type
        self.num_classes = num_classes

        if self.num_classes == 1:
            self.final = nn.Sigmoid()
        else:
            self.final = None

        channel = 64
        # 64 128 256 512 1024
        filters = [channel, channel * 2, channel * 4, channel * 8, channel * 16]

        self.Up4, self.Up3, self.Up2, self.Up1 = get_head(head_up, filters)

        if self.backbone_type == "resnet50":
            self.adjustment_block = AdjustmentBlock(64, 64)
            self.adjustment_block1 = AdjustmentBlock(256, 128)
            self.adjustment_block2 = AdjustmentBlock(512, 256)
            self.adjustment_block3 = AdjustmentBlock(1024, 512)
            self.adjustment_block4 = AdjustmentBlock(2048, 1024)

        self.pred = nn.Sequential(
            nn.Conv2d(filters[0], filters[0] // 2, kernel_size=1),
            nn.BatchNorm2d(filters[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0] // 2, num_classes, kernel_size=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        [e1, e2, e3, e4, e5] = inputs  # 64 256 512 1024 2048
        if self.backbone_type == "resnet50":
            e1 = self.adjustment_block(e1)
            e2 = self.adjustment_block1(e2)
            e3 = self.adjustment_block2(e3)
            e4 = self.adjustment_block3(e4)
            e5 = self.adjustment_block4(e5)
        # decoder
        d4 = self.Up4(e5, e4)  # (1,512,32,32)
        d3 = self.Up3(d4, e3)  # (1,256,64,64)
        d2 = self.Up2(d3, e2)  # (1,128,128,128)
        d1 = self.Up1(d2, e1)  # (1,64,256,256)
        pred = self.pred(d1)

        if self.final is not None:
            pred = self.final(pred)

        return pred


if __name__ == "__main__":
    print("head.py")

    # backbone = resnet50(pretrained=False)
    # model = Head(backbone_type='resnet50', num_classes=9, head_up="JUp_Concat")
    backbone = UnetEncoder()
    model = Head(backbone_type='base', num_classes=9, head_up="JUp_Concat")
    # ['UnetUp', 'JUp', 'JUp_Concat']
    # print(model)

    input_data = torch.randn((1, 3, 256, 256))
    output_backbone = backbone(input_data)
    output_data = model(output_backbone)

    print(output_data.shape)

    print(f'The parameters size of model is ' 
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')
