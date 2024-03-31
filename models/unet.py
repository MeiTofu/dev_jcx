#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 21:11
@Message: null
"""
import torch
from torch import nn

from models.head import Head
from models.resnet import resnet50
from models.vgg import VGG16


class Unet(nn.Module):
    def __init__(self, backbone_type="resnet50", num_classes=7, pretrained=False):
        super(Unet, self).__init__()
        self.backbone_type = backbone_type
        self.num_classes = num_classes

        if backbone_type == 'vgg':
            self.backbone = VGG16(pretrained=pretrained)
        elif backbone_type == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)

        self.head = Head(backbone_type=backbone_type, num_classes=num_classes)

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
        output_backbone = self.backbone(inputs)
        output_head = self.head(output_backbone)

        return output_head


if __name__ == "__main__":
    print("dev unet")

    model = Unet(backbone_type="resnet50", num_classes=7, pretrained=False)
    print(model)

    input_data = torch.randn((1, 3, 512, 512))
    output_data = model(input_data)

    print(output_data.shape)
