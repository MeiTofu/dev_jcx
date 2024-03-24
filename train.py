#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 22:12
@Message: null
"""
import argparse
import os.path
import time
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.unet import Unet
from utils.dataloader import UDataset, unet_dataset_collate
from utils.loss import Focal_Loss, CE_Loss
from utils.util import generate_dir

if __name__ == '__main__':
    print("train")
    parser = argparse.ArgumentParser(description='Unet Train Info')
    parser.add_argument('--info', default=None,
                        help='模型修改备注信息')
    parser.add_argument('--save_path', default='run',
                        help='训练信息保存路径')
    parser.add_argument('--seed', default=42,
                        help='random seed')
    parser.add_argument('--device', default='cuda:0',
                        help='device')
    parser.add_argument('--input_size', default=(512, 512),
                        help='the model input image size')
    parser.add_argument('--best_acc', default=0.5,
                        help='best_acc.')

    # 网络结构参数
    parser.add_argument('--backbone_type', type=str, default="resnet50",
                        help='选择主干网络')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='目标类别数，对应网络的输出特征通道数')
    parser.add_argument('--pretrain_backbone', type=bool, default=False,
                        help='主干网络是否加载预训练权重')
    parser.add_argument('--weight_path', default=None,
                        help='pre-training model load path.')

    # 加载数据相关参数
    parser.add_argument('--dataset_path', type=str, default="data/voc_diabetic",
                        help='数据集路径')
    parser.add_argument('--train_txt_path', type=str, default="data/voc_diabetic/ImageSets/Segmentation",
                        help='train/val 划分的 txt 文件路径')
    parser.add_argument('--mode', type=bool, default=True,
                        help='当前网络的训练模式：train/val')

    # 训练参数
    parser.add_argument('--epoch', type=int, default=80,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', default=4,
                        help='batch size when training.')
    parser.add_argument('--num_workers', default=8,
                        help='load data num_workers when training.')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate, 0.0005 is the default value for training')
    parser.add_argument('--lr_steps', default=[40, 70], type=list,
                        help='decrease lr every step-size epochs for MultiStepLR')
    parser.add_argument('--lr_gamma', default=0.2, type=float,
                        help='decrease lr by a factor of lr_gamma')

    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 实例化模型
    model = Unet(backbone_type=args.backbone_type, num_classes=args.num_classes, pretrained=args.pretrain_backbone)
    model.to(device)
    if args.weight_path is not None:
        model.load_state_dict(torch.load(args.weight_path, map_location=lambda storage, loc: storage), strict=False)
        print("已加载预训练权重：{}".format(args.weight_path))

    # 加载数据集
    cls_weights = np.ones([args.num_classes], np.float32)
    # 读取数据集对应的txt
    with open(os.path.join(args.train_txt_path, "train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.train_txt_path, "val.txt"), "r") as f:
        val_lines = f.readlines()

    dataset = {'train': UDataset(train_lines, num_classes=args.num_classes, train=True, dataset_path=args.dataset_path),
               'val': UDataset(val_lines, num_classes=args.num_classes, train=False, dataset_path=args.dataset_path)}
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=unet_dataset_collate, pin_memory=False),
        'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                          collate_fn=unet_dataset_collate, pin_memory=False)}
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}

    print('training dataset loaded with length : {}'.format(len(dataset['train'])))
    print('validation dataset loaded with length : {}'.format(len(dataset['val'])))

    # 定义优化器
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'betas': (0.9, 0.998), 'eps': 1e-8,
                                   'weight_decay': args.weight_decay}])
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 保存运行相关的参数信息
    current_time = time.strftime("%Y%m%d-%H_%M", time.localtime())
    save_dir = os.path.join(args.save_path, current_time)
    train_log_file = os.path.join(save_dir, '{}.txt'.format(current_time))
    save_info_dir = save_dir
    generate_dir(save_dir)
    # 读取配置文件
    argsDict = args.__dict__
    with open(os.path.join(save_dir, 'train_config.txt'), 'w', encoding='utf8') as f:
        f.write("---------training configuration information---------\n")
        f.write("current time : {}.\n".format(current_time))
        f.writelines('dataset train size : {}, val size : {}.\n\n'.format(len(dataset['train']), len(dataset['val'])))

        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- succeed -------------------')

    # 训练模型
    start_time = time.time()
    for epoch in range(args.epoch):
        print('-' * 38)
        print('Epoch {}/{}'.format(epoch, args.epoch - 1))
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('current lpr optimizer of lr:{}'.format(current_lr))

        # every epoch has train and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # lr_scheduler.step()
                model.train()  # set train model
            else:
                model.eval()  # set evaluate model

            running_loss = 0.0
            running_Acc = 0
            for batch in tqdm(dataloader[phase], desc=phase + ' Data Processing Progress'):
                imgs, pngs, labels = batch
                imgs = imgs.to(device)
                pngs = pngs.to(device)
                # labels.to(device)

                # clear gradient
                optimizer.zero_grad()

                # forward propagation, track history if only train
                with torch.set_grad_enabled(phase == 'train'):
                    weights = torch.from_numpy(cls_weights).to(device)
                    outputs = model(imgs)

                    # 计算损失
                    loss = CE_Loss(outputs, pngs, weights, num_classes=args.num_classes)

                    # backward and optimized only in the training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        pass
                    # TODO 评估

                # statistics loss
                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]

            # save data / learning rate regulator
            if phase == 'train':
                exp_lr_scheduler.step()
            else:
                pass

            print('{} Loss:{:.8f}'.format(phase, loss))
            with open(train_log_file, 'a', encoding='utf8') as f:
                f.write('Epoch {}/{},\tphase:{}\ncurrent_lr:{:.6f},\tloss:{:.6f},\t\n\n'.format(
                    epoch, args.epoch-1, phase, current_lr, epoch_loss))

            # save weight
            if phase == 'val' and epoch == args.epoch - 1:
                torch.save(model.state_dict(), os.path.join(save_dir, 'last_loss_{:.4f}.pth'.format(epoch_loss)))

    time_spend = time.time() - start_time
    # with open(train_log_file, 'a', encoding='utf8') as f:
    #     f.write('\nThe total time spent training the model is {:.0f}h{:.0f}m{:.0f}\n'.format(
    #         time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))
    #     f.write('Best val epoch : {}\n'.format(best_epoch))
    #     f.write('Best val Acc : {:4f}'.format(args.best_acc))
    #     f.write("\n\ntrain_loss:")
    #     f.write(",".join(str(round(_, 5)) for _ in loss_list))
    #     f.write("\nval_loss:")
    #     f.write(",".join(str(round(_, 5)) for _ in val_loss_list))

    print('\nThe total time spent training the model is {:.0f}h{:.0f}m{:.0f}.'.format(
        time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))
