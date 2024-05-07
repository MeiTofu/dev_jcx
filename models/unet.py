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
from models.unet_base import UnetEncoder


class Unet(nn.Module):
    def __init__(self, backbone_type="resnet50", num_channel=3, num_classes=9, pretrained=True, head_up="UnetUp"):
        super(Unet, self).__init__()
        self.backbone_type = backbone_type
        self.num_classes = num_classes

        if backbone_type == "resnet50":
            self.backbone = resnet50(num_channel=num_channel, pretrained=pretrained)
        elif backbone_type == "base":
            self.backbone = UnetEncoder(num_channel=num_channel)
        else:
            raise ValueError('Unsupported backbone type with `{}`, use base or resnet50.'.format(backbone_type))
        self.head = Head(backbone_type=backbone_type, num_classes=num_classes, head_up=head_up)

        if not pretrained:
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

    def freeze_backbone(self, freeze=False):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze



if __name__ == "__main__":
    print("unet.py")

    model = Unet(backbone_type="base", num_channel=3, num_classes=9, pretrained=False, head_up="JUp_Concat")
    # model = Unet(backbone_type="resnet50", num_channel=3, num_classes=9, pretrained=False)
    print(model)

    model.freeze_backbone(True)
    input_data = torch.randn((1, 3, 256, 256))
    output_data = model(input_data)
    model.freeze_backbone(False)

    print(output_data.shape)
    print(f'The parameters size of model is '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')

