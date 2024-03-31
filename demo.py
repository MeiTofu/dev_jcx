#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 19:32
@Message: null
"""
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt

from utils.util import generate_save_epoch

if __name__ == "__main__":
    print("demo")
    # raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # 保存模型权重epoch
    epochs = 150
    epoch_list = generate_save_epoch(epochs)
    print(epoch_list)
    # epoch_list = generate_save_epoch(epochs, period=20)
    # print(epoch_list)

    for epoch in range(epochs):
        if epoch in epoch_list:
            print(epoch, "eval model")

    # # 生成有序数组
    # arr = torch.arange(2 * 3 * 4)
    #
    # # 将数组reshape为维度为(2, 3, 4)
    # arr = arr.reshape(2, 3, 4)
    # print(arr)
    #
    # tp1 = torch.sum(arr, axis=[0,1])
    # print(tp1)
    # tp = torch.sum(arr, dim=(0, 1))
    # print(tp)
    # max_epoch = 20
    # iters = 200
    # model = resnet18(pretrained=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # mode = 'cosineAnnWarm'
    # if mode == 'cosineAnn':
    #     scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    # elif mode == 'cosineAnnWarm':
    #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max_epoch, T_mult=1)
    #     '''
    #     以T_0=5, T_mult=1为例:
    #     T_0:学习率第一次回到初始值的epoch位置.
    #     T_mult:这个控制了学习率回升的速度
    #         - 如果T_mult=1,则学习率在T_0,2*T_0,3*T_0,....,i*T_0,....处回到最大值(初始学习率)
    #             - 5,10,15,20,25,.......处回到最大值
    #         - 如果T_mult>1,则学习率在T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,.....,(1+T_mult+T_mult**2+...+T_0**i)*T0,处回到最大值
    #             - 5,15,35,75,155,.......处回到最大值
    #     example:
    #         T_0=5, T_mult=1
    #     '''
    # plt.figure()
    # cur_lr_list = []
    # for epoch in range(max_epoch):
    #     for batch in range(iters):
    #         '''
    #         这里scheduler.step(epoch + batch / iters)的理解如下,如果是一个epoch结束后再.step
    #         那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
    #         则可以在这里进行.step
    #         '''
    #         # scheduler.step(epoch + batch / iters)
    #         optimizer.step()
    #     scheduler.step()
    #     cur_lr = optimizer.param_groups[-1]['lr']
    #     cur_lr_list.append(cur_lr)
    #     # print('cur_lr:', cur_lr)
    # temp_lsit = cur_lr_list[:-1]
    # x_list = list(range(len(temp_lsit)))
    # plt.plot(x_list, temp_lsit)
    # plt.show()
