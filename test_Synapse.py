#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Jeff
@Description:
Based on https://github.com/McGregorWwww/UDTransNet
"""
import argparse

import numpy as np
from PIL import Image
from medpy import metric
import torch.optim
from scipy.ndimage import zoom
from torchvision.transforms import functional as F
import time

from tqdm import tqdm
import os
import cv2

from models.unet import Unet
from models.unet_base import UnetBase


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        iou = metric.binary.jc(pred, gt)
        return dice, iou
    elif pred.sum()==0 and gt.sum()==0:
        return 1, 1
    else:
        return 0, 0


def show_image_with_dice(predict_save, labs, n_labels=9):
    tmp_lbl = labs.astype(np.float32)
    tmp_3dunet = predict_save.astype(np.float32)
    metric_list = []
    for i in range(1, n_labels):
        metric_list.append(calculate_metric_percase(tmp_3dunet == i, tmp_lbl == i))
    metric_list = np.array(metric_list)

    dice_pred = np.mean(metric_list, axis=0)[0]
    iou_pred = np.mean(metric_list, axis=0)[1]
    dice_class = metric_list[:,0]

    return dice_pred, iou_pred, dice_class


def vis_save_synapse(input_img, pred, mask, save_path):
    blue   = [30,144,255] # aorta
    green  = [0,255,0]    # gallbladder
    red    = [255,0,0]    # left kidney
    cyan   = [0,255,255]  # right kidney
    pink   = [255,0,255]  # liver
    yellow = [255,255,0]  # pancreas
    purple = [128,0,255]  # spleen
    orange = [255,128,0]  # stomach
    if len(np.unique(mask)) > 2:
        # input_img = input_img.convert('RGB')
        if pred is not None:
            pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
            input_img = np.where(pred==1, np.full_like(input_img, blue  ), input_img)
            input_img = np.where(pred==2, np.full_like(input_img, green ), input_img)
            input_img = np.where(pred==3, np.full_like(input_img, red   ), input_img)
            input_img = np.where(pred==4, np.full_like(input_img, cyan  ), input_img)
            input_img = np.where(pred==5, np.full_like(input_img, pink  ), input_img)
            input_img = np.where(pred==6, np.full_like(input_img, yellow), input_img)
            input_img = np.where(pred==7, np.full_like(input_img, purple), input_img)
            input_img = np.where(pred==8, np.full_like(input_img, orange), input_img)
        else:
            # mask = mask.convert('RGB')
            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            input_img = np.where(mask==1, np.full_like(input_img, blue  ), input_img)
            input_img = np.where(mask==2, np.full_like(input_img, green ), input_img)
            input_img = np.where(mask==3, np.full_like(input_img, red   ), input_img)
            input_img = np.where(mask==4, np.full_like(input_img, cyan  ), input_img)
            input_img = np.where(mask==5, np.full_like(input_img, pink  ), input_img)
            input_img = np.where(mask==6, np.full_like(input_img, yellow), input_img)
            input_img = np.where(mask==7, np.full_like(input_img, purple), input_img)
            input_img = np.where(mask==8, np.full_like(input_img, orange), input_img)

        input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, input_img)


def test_Synapse(test_model, image, label, vis_save_path=None, save=False):
    dice_pred_all = np.zeros(1)
    iou_pred_all = np.zeros(1)
    dice_class_all = np.zeros((1, 8))

    res_vis = []
    dice_pred, iou_pred, dice_class = [],[],[]
    output_size = args.input_size

    # origin size for visualization
    input_512, label_512 = image, label # cv2
    label_512 = np.array(label_512, np.uint8)
    # resize for model input
    image_cv = cv2.resize(image, output_size)
    label_cv = cv2.resize(label, output_size)
    image_pil, label_pil = F.to_pil_image(image_cv), F.to_pil_image(label_cv)

    # w, h = input_512.size
    # image = zoom(image, output_size[0] / w, order=1)
    # label = zoom(label, (output_size[0] / w, output_size[1] / h), order=0)

    image_tensor = F.to_tensor(image_pil).unsqueeze(0)
    label_np = np.array(label_pil, np.uint8)

    output = test_model(image_tensor.to(device))
    predict_save = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
    predict_save = predict_save.cpu().data.numpy()
    res_vis.append(output)
    dice_pred_tmp, iou_tmp, dice_class_tmp = show_image_with_dice(predict_save, label_np, n_labels=args.num_classes)
    dice_pred.append(dice_pred_tmp)
    iou_pred.append(iou_tmp)
    dice_class.append(dice_class_tmp)

    if save:
        res_vis = torch.cat(res_vis, dim=0)
        predict_save = torch.argmax(torch.softmax(res_vis.mean(0), dim=0), dim=0).cpu().data.numpy().astype(np.uint8)
        predict_save_512 = zoom(predict_save, (512 / output_size[0], 512 / output_size[1]), order=0)
        vis_save_synapse(input_512, predict_save_512, label_512, save_path=vis_save_path+'_'+'_unet'+'.jpg')
        vis_save_synapse(input_512, None, label_512, save_path=vis_save_path+'_'+'_gt.jpg')
    dice_pred_all += np.array(dice_pred)
    iou_pred_all += np.array(iou_pred)
    dice_class_all += np.array(dice_class)

    return dice_pred_all, iou_pred_all, dice_class_all


if __name__ == '__main__':
    print("test.py")
    parser = argparse.ArgumentParser(description='Unet Test Info')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--input_size', default=(256, 256), help='the model input image size')
    parser.add_argument('--num_classes', type=int, default=9, help='目标类别数，对应网络的输出特征通道数')
    parser.add_argument('--backbone_type', type=str, default="base", help='选择主干网络')
    parser.add_argument('--head_up', type=str, default="JUp_Concat", help='可选: [UnetUp, JUp_Concat]')
    parser.add_argument('--weight_path',
                        default="/home/amax/Jiao/experiment/train_dev_Synapse/20240507-19_56/best_model_weight.pth",
                        help='加载训练好的模型权重')
    parser.add_argument('--dataset_path', type=str, default="../dataset/Synapse_all",
                        help='训练数据集路径')
    parser.add_argument('--test_txt_path', type=str, default="test.txt",
                        help='val 划分的 txt 文件路径')
    parser.add_argument('--save_path', default="Synapse_visualize_test/", help='评估结果可视化保存路径')
    parser.add_argument('--save', type=bool, default=False, help='保存评估结果')
    args = parser.parse_args()

    device = torch.device(args.device)
    # 实例化模型
    model = Unet(args.backbone_type, num_classes=args.num_classes, head_up=args.head_up)
    # model = UnetBase(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    print("loaded model weight: ", args.weight_path)
    model.to(device)
    model.eval()

    with open(os.path.join(args.dataset_path, args.test_txt_path), "r") as f:
        val_lines = f.readlines()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dice_pred = np.zeros(1)
    iou_pred = np.zeros(1)
    dice_class = np.zeros((1, 8))
    dice_ens = 0.0
    dice_5folds, iou_5folds = [], []
    test_count = 0
    end = time.time()
    for i, line in enumerate(tqdm(val_lines, desc="Model Test"), 1):
        line_name = line.split()[0]
        file_prefix, file_suffix = os.path.splitext(line_name)
        image_path = os.path.join(args.dataset_path, 'images/'+line_name)
        mask_path = os.path.join(args.dataset_path, 'masks/'+file_prefix+".png")
        test_data = cv2.imread(image_path)
        test_label = cv2.imread(mask_path, 0)
        vision_save_path = args.save_path + str(i)
        dice_pred_t, iou_pred_t, dice_class_t = test_Synapse(model, test_data, test_label, vision_save_path, save=args.save)

        dice_pred_t = np.array(dice_pred_t)
        iou_pred_t = np.array(iou_pred_t)

        dice_pred += dice_pred_t
        iou_pred += iou_pred_t

        dice_class_t = np.array(dice_class_t)
        dice_class += dice_class_t
        test_count = i

    inference_time = (time.time() - end)
    print("inference_time", inference_time)
    dice_pred = dice_pred / test_count * 100.0
    iou_pred = iou_pred / test_count * 100.0
    dice_class = dice_class / test_count * 100.0
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    print("Avg Dice:", dice_pred)
    print("Avg IoU:", iou_pred)
    dice_pred_mean = dice_pred.mean()
    iou_pred_mean = iou_pred.mean()

    dice_class_mean = dice_class.mean(0)
    dice_pred_std = np.std(dice_pred, ddof=1)
    iou_pred_std = np.std(iou_pred, ddof=1)
    print("dice: {:.2f}+{:.2f}".format(dice_pred_mean, dice_pred_std))
    print("iou: {:.2f}+{:.2f}".format(iou_pred_mean, iou_pred_std))
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    print("dice class:", dice_class_mean)