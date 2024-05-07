#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Jeff
@Description:
"""
import argparse
import os.path
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.unet import Unet
from models.unet_base import UnetBase
from utils.dataloader_Synapse import ValGenerator, ImageToImage2D_kfold, RandomGenerator
from utils.loss import MultiClassDiceCE, MultiClassDiceFocal
from utils.util import generate_dir, set_random_seed, save_info_when_training, set_optimizer_lr, \
    save_dice_when_training


def main(args, model, dataloader, criterion, log_dir):
    # 获取当前模型的设备
    device = next(model.parameters()).device
    # 根据当前batch_size，自适应调整学习率
    if train_sizes['train'] > 5000:
        nbs = 16
        lr_limit_max = 1e-4 if args.optimizer_type in ['Adam', 'AdamW'] else 1e-1
        lr_limit_min = 1e-4 if args.optimizer_type in ['Adam', 'AdamW'] else 5e-4
        init_lr_fit = min(max(args.batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit = min(max(args.batch_size / nbs * (args.init_lr * args.min_lr_factor), lr_limit_min * 1e-2),
                         lr_limit_max * 1e-2)
    else:
        init_lr_fit = args.init_lr
        min_lr_fit = args.init_lr * args.min_lr_factor

    # 定义优化器
    optimizer = {'Adam': torch.optim.Adam(model.parameters(), init_lr_fit, betas=(0.9, 0.999), weight_decay=args.weight_decay),
                 'AdamW': torch.optim.AdamW(params=model.parameters(), lr=init_lr_fit, weight_decay=args.weight_decay)
                 }[args.optimizer_type]

    # 每个epoch的训练信息
    loss_list, val_loss_list, train_dice_list, val_dice_list, lr_list = [], [], [], [], []
    best_epoch = 0

    # 训练模型
    start_time = time.time()
    for epoch in range(args.epoch):
        print('-' * 38)
        print('Epoch {}/{}'.format(epoch, args.epoch - 1))
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_lr)
        print('current model optimizer of lr:{}'.format(current_lr))

        # 解冻模型主干权重
        if args.freeze_train and epoch >= args.freeze_epoch:
            model.freeze_backbone(False)
            args.freeze_train = False
            print("已解冻所有模型权重")

        # every epoch has train and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # lr_scheduler.step()
                # set train model
                model.train()
            else:
                model.eval()

            running_loss = []
            dice_list = []
            loss_sum = 0
            dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
            for sampled_batch, names in tqdm(dataloader[phase], desc=phase + ' Data Processing Progress'):
                images, masks = sampled_batch['image'], sampled_batch['label']
                images, masks = images.to(device), masks.to(device)

                # forward propagation, track history if only train
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(images)
                    # 计算损失 默认使用 CE Loss + Dice Loss 组合
                    loss = criterion(preds, masks.long(), softmax=True)
                    train_dice = criterion.calculate_dice(preds, masks.long(), softmax=True)

                    # backward and optimized only in the training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics loss
                for n in range(len(images)):
                    running_loss.append(loss.item())
                    dice_list.append(train_dice.item())

                loss_sum += len(images) * loss.item()
                dice_sum += len(images) * train_dice.item()

                # if i == len(dataloader[phase]):
                #     average_loss = loss_sum / (args.batch_size * (i - 1) + len(images))
                #     train_dice_avg = dice_sum / (args.batch_size * (i - 1) + len(images))
                # else:
                #     average_loss = loss_sum / (i * args.batch_size)
                #     train_dice_avg = dice_sum / (i * args.batch_size)

            # 当前epoch 的loss dice
            epoch_loss = sum(running_loss) / len(running_loss)
            epoch_dice = sum(dice_list) / len(dice_list) if len(dice_list) > 0 else 0.0

            # save data / learning rate regulator
            if phase == 'train':
                loss_list.append(epoch_loss)
                train_dice_list.append(epoch_dice)
                print('{} Loss:{:.8f}, dice:{:.4f} '.format(phase, epoch_loss, epoch_dice))
            else:
                val_loss_list.append(epoch_loss)
                val_dice_list.append(epoch_dice)
                print('{} Loss:{:.8f} dice:{:.4f} '.format(phase, epoch_loss, epoch_dice))
                # 保存当前最好模型权重
                if args.best_acc <= epoch_dice:
                    args.best_acc = epoch_dice
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(log_dir, 'best_model_weight.pth'))
                    print("已保存当前最好模型权重：", os.path.join(log_dir, 'best_model_weight.pth'))

                save_info_when_training(loss_list, val_loss_list, lr_list, log_dir)
                save_dice_when_training(train_dice_list, val_dice_list, log_dir)

            with open(os.path.join(log_dir, "epoch_loss.txt"), 'a', encoding='utf8') as f:
                if phase == 'train':
                    f.write('\n' + '-' * 42 + '\n')
                f.write('Epoch {}/{},\tphase:{}\ncurrent_lr:{:.6f},\tloss:{:.6f}, \tdice:{:.4f}\n'
                        .format(epoch, args.epoch - 1, phase, current_lr, epoch_loss, epoch_dice))

            # 保存最后的模型权重，并绘制训练曲线
            if phase == 'val' and epoch == args.epoch - 1:
                torch.save(model.state_dict(), os.path.join(log_dir, '{}_loss{:.4f}_dice{:.4f}.pth'
                                                            .format(epoch, epoch_loss, epoch_dice)))

        # 调整当前训练的学习率
        set_optimizer_lr(optimizer, epoch, args.epoch, warmup_epochs=args.freeze_epoch // 2, max_lr=init_lr_fit, min_lr=min_lr_fit)

    # 最后保存模型训练信息
    time_spend = time.time() - start_time
    print('\nThe save best weight epoch {}, dice={}.'.format(best_epoch, args.best_acc))
    print('\nThe total time spent training the model is {:.0f}h{:.0f}m{:.0f}s.'.format(
        time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))
    with open(os.path.join(log_dir, "epoch_loss.txt"), 'a', encoding='utf8') as f:
        f.write('\nThe save best weight epoch {}, dice={}.'.format(best_epoch, args.best_acc))
        f.write('\nThe total time spent training the model is {:.0f}h{:.0f}m{:.0f}s.\n'.format(
            time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))
        f.write("\ntrain_loss:")
        f.write(",".join(str(round(_, 4)) for _ in loss_list))
        f.write("\nval_loss:")
        f.write(",".join(str(round(_, 4)) for _ in val_loss_list))

        f.write("\ntrain dice:")
        f.write(",".join(str(round(_, 4)) for _ in train_dice_list))
        f.write("\nval dice:")
        f.write(",".join(str(round(_, 4)) for _ in val_dice_list))


def build_model(args, device):
    """
    创建模型，加载预训练权重并冻结主干
    :param args: 模型配置参数
    :param device: 推理设备
    :return:
    """
    # 实例化模型
    model = Unet(backbone_type=args.backbone_type, num_classes=args.num_classes, head_up=args.head_up)
    # model = UnetBase(num_classes=args.num_classes)
    model.to(device)
    if args.weight_path is not None:
        print("Model weights loading：{}".format(args.weight_path))
        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.weight_path, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 显示没有匹配上的Key
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("Fail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # 训练冻结模型主干权重
    if args.freeze_train:
        model.freeze_backbone(True)
        print("已冻结主干部分权重")

    return model


def build_dataloader(args):
    """
    创建数据集加载器
    :param args: 数据集加载器配置参数
    :return:
    """
    # 读取数据集对应的txt
    with open(os.path.join(args.dataset_path, args.val_txt_path), "r") as f:
        val_lines = f.readlines()
    with open(os.path.join(args.dataset_path, args.train_txt_path), "r") as f:
        train_lines = f.readlines()
    # 数据预处理的操作
    val_tf = ValGenerator(output_size=args.input_size)
    train_tf = RandomGenerator(output_size=args.input_size)
    dataset = {'train': ImageToImage2D_kfold(dataset_path=args.dataset_path, split_txt=train_lines, image_size=args.input_size[0], joint_transform=train_tf, split='train'),
               'val': ImageToImage2D_kfold(dataset_path=args.dataset_path, split_txt=val_lines, image_size=args.input_size[0], joint_transform=val_tf, split='val')}
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True),
        'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          pin_memory=True)}
    # 训练集/验证集的大小
    dataset_size = {x: len(dataset[x]) for x in ['train', 'val']}
    print("Dataset statistical results：")
    print('training dataset loaded with length : {}'.format(dataset_size['train']))
    print('validation dataset loaded with length : {}'.format(dataset_size['val']))

    return dataloader, dataset_size


def build_criterion(args):
    if args.focal_loss:
        criterion = MultiClassDiceFocal(num_classes=args.num_classes)
    else:
        criterion = MultiClassDiceCE(num_classes=args.num_classes)
    return criterion

def save_train_info(args, dataset_sizes):
    """
    保存模型训练参数信息
    :param args: 训练参数配置
    :param dataset_sizes: 数据集大小
    :return:
    """
    current_time = time.strftime("%Y%m%d-%H_%M", time.localtime())
    log_dir = os.path.join(args.log_path, current_time)
    generate_dir(log_dir)
    # 读取配置文件
    argsDict = args.__dict__
    with open(os.path.join(log_dir, 'train_config.txt'), 'w', encoding='utf8') as f:
        f.write("---------training configuration information---------\n")
        f.write("current time : {}.\n".format(current_time))
        f.writelines('dataset train size : {}, val size : {}.\n\n'.format(dataset_sizes['train'], dataset_sizes['val']))
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- ok -------------------')
    return log_dir


if __name__ == '__main__':
    print("train")
    # 设置随机种子，保证结果的可复现
    set_random_seed(42)
    parser = argparse.ArgumentParser(description='Unet Train Info')
    parser.add_argument('--info', default=['Base+JUp_Concat'],
                        help='模型修改备注信息')
    parser.add_argument('--log_path', default='../experiment/train_dev_Synapse',
                        help='训练信息保存路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--device', default='cuda:1',
                        help='device')
    parser.add_argument('--input_size', default=(256, 256),
                        help='the model input image size')
    parser.add_argument('--best_acc', default=0.2,
                        help='best_acc.')
    # ======================= 网络结构参数=============================
    parser.add_argument('--backbone_type', type=str, default="base",
                        help='选择主干网络')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='目标类别数，对应网络的输出特征通道数')
    parser.add_argument('--weight_path', default=None,
                        help='pre-training model load path.')
    parser.add_argument('--head_up', type=str, default="JUp_Concat",
                        help='选择头网络的多尺度特征融合方式，可选: [UnetUp, JUp, JUp_Concat]')
    # ======================= 加载数据相关参数=============================
    parser.add_argument('--dataset_path', type=str, default="../dataset/Synapse_all",
                        help='训练数据集路径')
    parser.add_argument('--train_txt_path', type=str, default="train.txt",
                        help='train 划分的 txt 文件路径')
    parser.add_argument('--val_txt_path', type=str, default="test.txt",
                        help='val 划分的 txt 文件路径')
    # ======================= 训练参数=============================
    parser.add_argument('--epoch', type=int, default=120,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', default=16,
                        help='batch size when training.')
    parser.add_argument('--num_workers', default=16,
                        help='load data num_workers when training.')
    parser.add_argument('--freeze_train', type=bool, default=False,
                        help='freeze backbone when start train.')
    parser.add_argument('--freeze_epoch', type=int, default=0,
                        help='模型冻结主干训练的epoch')
    parser.add_argument('--init_lr', default=0.0001, type=float,
                        help='初始学习率，0.0001 是训练的默认值')
    parser.add_argument('--min_lr_factor', default=0.01, type=float,
                        help='当使用 CosineAnnealingWarmRestarts 时的最小学习率')
    # ======================= 损失函数相关参数=============================
    parser.add_argument('--focal_loss', default=False, type=bool,
                        help='是否使用 Focal Loss 以防止正负样本不平衡，若不选择则默认使用 CE Loss')
    # ======================= 优化器相关参数=============================
    parser.add_argument('--optimizer_type', default='AdamW', type=str,
                        help='选择使用的优化器，可选(Adam, AdamW, SGD)')
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Adam 优化器的参数 weight decay')
    # ======================= 学习率调节器相关参数 =============================
    parser.add_argument('--scheduler_type', default='Cosine_Warmup', type=str,
                        help='选择使用的学习率调节器，可选(MultiStepLR, CosineAnnealingWarmRestarts)')
    parser.add_argument('--lr_steps', default=[10, 18], type=list,
                        help='当使用 MultiStepLR 时，在指定步长进行学习率调节')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='当使用 MultiStepLR 时，学习率调节系数，即当前学习率乘以该系数')
    train_args = parser.parse_args()

    # 选择训练模型使用的设备
    train_device = torch.device(train_args.device if torch.cuda.is_available() else "cpu")
    print("Inference device: ", torch.cuda.get_device_name(train_device))

    # 实例化模型，并加载预训练权重和冻结主干
    train_model = build_model(args=train_args, device=train_device)

    # 加载数据集
    train_dataloader, train_sizes = build_dataloader(args=train_args)

    # 构造损失函数
    train_criterion = build_criterion(args=train_args)

    # 保存运行相关的参数信息
    train_log_dir = save_train_info(args=train_args, dataset_sizes=train_sizes)

    # 开始训练
    main(
        args=train_args,
        model=train_model,
        dataloader=train_dataloader,
        criterion=train_criterion,
        log_dir=train_log_dir)
