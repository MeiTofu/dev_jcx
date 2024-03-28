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

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.unet import Unet
from utils.dataloader import UDataset, unet_dataset_collate
from utils.loss import Focal_Loss, CE_Loss, Dice_loss
from utils.util import generate_dir, NAME_CLASSES, set_random_seed, save_info_when_training
from utils.util_eval import Evaluator
from utils.util_metrics import f_score


def run():
    # 是否给不同种类赋予不同的损失权值，默认是平衡的。设置的话，注意设置成numpy形式的，长度和num_classes一样。
    # 例如 num_classes = 3，则有 cls_weights = np.array([1, 2, 3], np.float32)
    cls_weights = np.ones([args.num_classes], np.float32)

    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=unet_dataset_collate, pin_memory=False),
        'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                          collate_fn=unet_dataset_collate, pin_memory=False)}
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}

    print("Dataset statistical results：")
    print('training dataset loaded with length : {}'.format(len(dataset['train'])))
    print('validation dataset loaded with length : {}'.format(len(dataset['val'])))

    # 根据当前batch_size，自适应调整学习率
    if len(dataset['train']) > 500:
        nbs = 16
        lr_limit_max = 1e-4 if args.optimizer_type == 'Adam' else 1e-1
        lr_limit_min = 1e-4 if args.optimizer_type == 'Adam' else 5e-4
        init_lr_fit = min(max(args.batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit = min(max(args.batch_size / nbs * args.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    else:
        init_lr_fit = args.init_lr
        min_lr_fit = args.init_lr * 0.01
    # 定义优化器
    optimizer = {'Adam': torch.optim.Adam([{'params': model.parameters(), 'lr': init_lr_fit,
                                            'betas': (0.9, 0.998), 'weight_decay': args.weight_decay}]),
                 'AdamW': torch.optim.AdamW(params=model.parameters(), lr=init_lr_fit, weight_decay=args.weight_decay)
                 }[args.optimizer_type]

    # 学习率调节器
    exp_lr_scheduler = {
        'MultiStepLR': lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma),
        'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epoch, T_mult=1, eta_min=min_lr_fit)
    }[args.scheduler_type]

    # 保存运行相关的参数信息
    current_time = time.strftime("%Y%m%d-%H_%M", time.localtime())
    save_dir = os.path.join(args.save_path, current_time)
    train_log_file = os.path.join(save_dir, '{}.txt'.format(current_time))
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
        f.writelines('------------------- success -------------------')

    # 定义评估器
    evaluator = Evaluator(args.input_size, args.num_classes, device, image_lines=val_lines,
                          dataset_path=args.dataset_path, log_dir=save_dir, period=args.period)

    loss_list, val_loss_list, val_f_score_list, lr_list = [], [], [], []

    # 训练模型
    start_time = time.time()
    for epoch in range(args.epoch):
        print('-' * 38)
        print('Epoch {}/{}'.format(epoch, args.epoch - 1))
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_lr)
        print('current model optimizer of lr:{}'.format(current_lr))

        # every epoch has train and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # lr_scheduler.step()
                model.train()  # set train model
            else:
                model.eval()  # set evaluate model

            running_loss = 0.0
            running_score = 0.0
            for batch in tqdm(dataloader[phase], desc=phase + ' Data Processing Progress'):
                imgs, pngs, labels = batch
                imgs = imgs.to(device)
                pngs = pngs.to(device)
                labels = labels.to(device)

                # clear gradient
                optimizer.zero_grad()

                # forward propagation, track history if only train
                with torch.set_grad_enabled(phase == 'train'):
                    weights = torch.from_numpy(cls_weights).to(device)
                    outputs = model(imgs)
                    # 计算损失 默认使用 CE Loss + Dice Loss 组合
                    if args.focal_loss:
                        loss = Focal_Loss(outputs, pngs, weights, num_classes=args.num_classes)
                    else:
                        loss = CE_Loss(outputs, pngs, weights, num_classes=args.num_classes)

                    if args.dice_loss:
                        loss_dice = Dice_loss(outputs, labels)
                        loss = loss + loss_dice

                    # backward and optimized only in the training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        val_score = f_score(outputs, labels)
                        running_score += val_score

                # statistics loss
                running_loss += loss.item()

            epoch_loss = running_loss / (dataset_sizes[phase] // args.batch_size)
            epoch_score = running_score / (dataset_sizes[phase]//args.batch_size)

            # save data / learning rate regulator
            if phase == 'train':
                exp_lr_scheduler.step()
                loss_list.append(epoch_loss)
                print('{} Loss:{:.8f} '.format(phase, epoch_loss))
            else:
                # 计算当前 epoch 的 precise mIoU
                epoch_acc = evaluator.on_epoch_end(epoch, model_eval=model, classes_eval=NAME_CLASSES, draw_info=False)
                val_loss_list.append(epoch_loss)
                val_f_score_list.append(epoch_score.cpu().numpy())
                print('{} Loss:{:.8f} F_score:{:.4f}'.format(phase, epoch_loss, epoch_score))
                if args.best_acc < epoch_acc:
                    args.best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_weight.pth'))

            with open(train_log_file, 'a', encoding='utf8') as f:
                f.write('Epoch {}/{},\tphase:{}\ncurrent_lr:{:.6f},\tloss:{:.6f},\tf_score:{:.4f}\t\n\n'.format(
                    epoch, args.epoch-1, phase, current_lr, epoch_loss, epoch_score))

            # save weight
            if phase == 'val' and epoch == args.epoch - 1:
                torch.save(model.state_dict(), os.path.join(save_dir, 'last_loss_{:.4f}.pth'.format(epoch_loss)))
                save_info_when_training(loss_list, val_loss_list, val_f_score_list, lr_list, save_dir)

    time_spend = time.time() - start_time

    print('\nThe total time spent training the model is {:.0f}h{:.0f}m{:.0f}.'.format(
        time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))


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
    parser.add_argument('--num_classes', type=int, default=7,
                        help='目标类别数，对应网络的输出特征通道数')
    parser.add_argument('--pretrain_backbone', type=bool, default=False,
                        help='主干网络是否加载预训练权重')
    parser.add_argument('--weight_path', default="weight/init_unet.pth",
                        help='pre-training model load path.')

    # 加载数据相关参数
    parser.add_argument('--dataset_path', type=str, default="data/VOCdevkit",
                        help='数据集路径')
    parser.add_argument('--train_txt_path', type=str, default="data/VOCdevkit/ImageSets/Segmentation",
                        help='train/val 划分的 txt 文件路径')
    parser.add_argument('--mode', type=bool, default=True,
                        help='当前网络的训练模式：train/val')

    # 训练参数
    parser.add_argument('--epoch', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('--period', type=int, default=20,
                        help='Evaluated every interval epoch')
    parser.add_argument('--batch_size', default=4,
                        help='batch size when training.')
    parser.add_argument('--num_workers', default=8,
                        help='load data num_workers when training.')
    parser.add_argument('--init_lr', default=0.0005, type=float,
                        help='初始学习率，0.0005 是训练的默认值')

    # 损失函数相关参数
    parser.add_argument('--focal_loss', default=False, type=bool,
                        help='是否使用 Focal Loss 以防止正负样本不平衡，若不选择则默认使用 CE Loss')
    parser.add_argument('--dice_loss', default=True, type=bool,
                        help='是否使用 Dice Loss 根据label额外计算损失，使用后效果更好')
    # 优化器相关参数
    parser.add_argument('--optimizer_type', default='Adam', type=str,
                        help='选择使用的优化器，可选(Adam, AdamW, SGD)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Adam 优化器的参数 weight decay')
    # 学习率调节器相关参数
    parser.add_argument('--scheduler_type', default='CosineAnnealingWarmRestarts', type=str,
                        help='选择使用的学习率调节器，可选(MultiStepLR, CosineAnnealingWarmRestarts)')
    parser.add_argument('--lr_steps', default=[10, 18], type=list,
                        help='当使用 MultiStepLR 时，在指定步长进行学习率调节')
    parser.add_argument('--lr_gamma', default=0.2, type=float,
                        help='当使用 MultiStepLR 时，学习率调节系数，即当前学习率乘以该系数')
    parser.add_argument('--min_lr', default=0.000005, type=float,
                        help='当使用 CosineAnnealingWarmRestarts 时的最小学习率')

    args = parser.parse_args()
    # 设置随机种子，保证结果的可复现
    set_random_seed(args.seed)
    # 选择训练模型使用的设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print(torch.cuda.get_device_name(device))

    # 实例化模型
    model = Unet(backbone_type=args.backbone_type, num_classes=args.num_classes, pretrained=args.pretrain_backbone)
    model.to(device)
    if args.weight_path is not None:
        model.load_state_dict(torch.load(args.weight_path, map_location=lambda storage, loc: storage), strict=True)
        print("已加载预训练权重：{}".format(args.weight_path))

    # 加载数据集
    # 读取数据集对应的txt
    with open(os.path.join(args.train_txt_path, "train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.train_txt_path, "val.txt"), "r") as f:
        val_lines = f.readlines()

    dataset = {'train': UDataset(train_lines, num_classes=args.num_classes, train=True, dataset_path=args.dataset_path),
               'val': UDataset(val_lines, num_classes=args.num_classes, train=False, dataset_path=args.dataset_path)}
    # 开始训练
    run()
