#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/4/6 21:42
@Message: null
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import Unet
from utils.dataloader import UDataset, unet_dataset_collate
from utils.loss import Focal_Loss, Dice_loss, CE_Loss
from utils.util_metrics import calculate_score

if __name__ == "__main__":
    print("inference")
    parser = argparse.ArgumentParser(description='Unet Inference Info')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--input_size', default=(512, 512), help='the model input image size')
    parser.add_argument('--backbone_type', type=str, default="resnet50", help='选择主干网络')
    parser.add_argument('--num_classes', type=int, default=7, help='目标类别数，对应网络的输出特征通道数')
    parser.add_argument('--weight_path', default="weight/init_unet.pth", help='加载训练好的模型权重')
    parser.add_argument('--dataset_path', type=str, default="data/voc_dev", help='数据集路径')
    parser.add_argument('--train_txt_path', type=str, default="data/voc_dev/ImageSets/Segmentation/test.txt",
                        help='train/val 划分的 txt 文件路径')
    parser.add_argument('--batch_size', default=1,
                        help='batch size when training.')
    parser.add_argument('--num_workers', default=2,
                        help='load data num_workers when training.')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 实例化UNet网络
    unet = Unet(backbone_type=args.backbone_type, num_classes=args.num_classes)
    unet.to(device)
    if args.weight_path is not None:
        unet.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=True)
        print("已加载预训练权重：{}".format(args.weight_path))
    unet.eval()

    with open(os.path.join(args.train_txt_path), "r") as f:
        val_lines = f.readlines()

    dataset = UDataset(val_lines, num_classes=args.num_classes, train=False, dataset_path=args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=unet_dataset_collate, pin_memory=False)
    print('dataset loaded with length : {}'.format(len(dataset)))

    cls_weights = np.ones([args.num_classes], np.float32)
    # 推理阶段
    for batch in tqdm(dataloader, desc='Inference Data Processing Progress'):
        imgs, pngs, labels = batch
        imgs = imgs.to(device)
        pngs = pngs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights).to(device)
            outputs = unet(imgs)
            # 计算损失
            loss_focal = Focal_Loss(outputs, pngs, weights, num_classes=args.num_classes)
            loss_ce = CE_Loss(outputs, pngs, weights, num_classes=args.num_classes)
            loss_dice = Dice_loss(outputs, labels)
            # 计算指标 dice_coefficient, mIoU, precision, recall
            dice_coefficient, mIoU, precision, recall = calculate_score(outputs, labels)

            print()


