#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 23:03
@Message: null
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


# def CE_Loss(inputs, target, cls_weights, num_classes=21):
#     n, c, h, w = inputs.size()
#     nt, ht, wt = target.size()
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
#
#     temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#     temp_target = target.view(-1)
#
#     CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
#     return CE_loss


# def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
#     n, c, h, w = inputs.size()
#     nt, ht, wt = target.size()
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
#
#     temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#     temp_target = target.view(-1)
#
#     logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
#                                                                                                  temp_target)
#     pt = torch.exp(logpt)
#     if alpha is not None:
#         logpt *= alpha
#     loss = -((1 - pt) ** gamma) * logpt
#     loss = loss.mean()
#     return loss


# def Dice_loss(inputs, target, beta=1, smooth=1e-5):
#     n, c, h, w = inputs.size()
#     nt, ht, wt, ct = target.size()
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
#
#     temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
#     temp_target = target.view(n, -1, ct)
#
#     # --------------------------------------------#
#     #   计算dice loss
#     # --------------------------------------------#
#     tp = torch.sum(temp_target[..., :-1] * temp_inputs, dim=[0, 1])
#     fp = torch.sum(temp_inputs, dim=[0, 1]) - tp
#     fn = torch.sum(temp_target[..., :-1], dim=[0, 1]) - tp
#
#     score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
#     dice_loss = 1 - torch.mean(score)
#     return dice_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class MultiClassDiceCE(nn.Module):
    def __init__(self, num_classes=9):
        super(MultiClassDiceCE, self).__init__()
        self.CE_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.CE_weight = 0.5
        self.dice_weight = 0.5

    def calculate_dice(self, inputs, targets, softmax=True):
        dice = 1.0 - self.dice_loss(inputs, targets,softmax=softmax)
        return dice

    def forward(self, inputs, targets, softmax=True):
        dice = self.dice_loss(inputs, targets,softmax=softmax)
        CE = self.CE_loss(inputs, targets)
        dice_CE_loss = self.dice_weight * dice + self.CE_weight * CE
        return dice_CE_loss


class MultiClassDiceFocal(nn.Module):
    def __init__(self, num_classes=9):
        super(MultiClassDiceFocal, self).__init__()
        self.focal_loss = FocalLoss(num_classes)
        self.dice_loss = DiceLoss(num_classes)
        self.focal_weight = 0.5
        self.dice_weight = 0.5

    def calculate_dice(self, inputs, targets, softmax=True):
        dice = 1.0 - self.dice_loss(inputs, targets,softmax=softmax)
        return dice

    def forward(self, inputs, targets, softmax=True):
        dice = self.dice_loss(inputs, targets,softmax=softmax)
        focal = self.focal_loss(inputs, targets)
        dice_focal_loss = self.dice_weight * dice + self.focal_weight * focal
        return dice_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, num_classes=9, alpha=0.25, gamma=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

        # print('Focal Loss:')
        # print('    Alpha = {}'.format(self.alpha))
        # print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

if __name__ == "__main__":
    print("loss")

    print("success")
