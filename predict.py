import argparse
import copy
import os

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from models.unet import Unet
from utils.util import cvtColor, preprocess_input, resize_image, add_info_to_image
from utils.util import CLASSES as name_classes, COLORS_CV as colors


def statistical_segmentation_results(prs, total_classes, original_size):
    """
    统计分割结果中每种类别的像素占比
    :param prs: 对应预测的每种类别的像素的 np 类型数据
    :param total_classes: 所有的类别数
    :param original_size: 原输入图像的尺寸 (w, h)
    :return:
    """
    classes_nums = np.zeros([total_classes])
    total_points_num = original_size[0] * original_size[1]
    print('-' * 64)
    print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
    print('-' * 64)
    for i in range(total_classes):
        num = np.sum(prs == i)
        ratio = num / total_points_num * 100
        if num > 0:
            print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
            print('-' * 64)
        classes_nums[i] = num
    print("classes_nums:", classes_nums)


def show_segmentation_results(origi_img, prs, total_classes, label_info=True):
    """
    将分割预测的结果在原图中显示
    :param origi_img: 原输入图像
    :param prs: 对应预测的每种类别的像素的 np 类型数据
    :param total_classes: 所有的类别数
    :param label_info: 是否显示标签信息，默认即可
    :return: 包含分割结果和标签的 PIL 类型数据
    """
    seg_img = np.zeros((np.shape(prs)[0], np.shape(prs)[1], 3))
    for c in range(total_classes):
        seg_img[:, :, 0] += ((prs[:, :] == c) * colors[c][0]).astype('uint8')
        seg_img[:, :, 1] += ((prs[:, :] == c) * colors[c][1]).astype('uint8')
        seg_img[:, :, 2] += ((prs[:, :] == c) * colors[c][2]).astype('uint8')

    my_image = np.array(origi_img)
    # 将预测结果不为背景（ > 0）的像素点设为白色(255, 255, 255) => 方便目标区域着色
    my_image[:, :, :][seg_img[:, :, :] > 0] = 255
    show_image = Image.fromarray(np.uint8(my_image))
    if label_info:
        # 添加对应标签信息
        show_image = add_info_to_image(show_image)
    return show_image


if __name__ == "__main__":
    print("predict")
    parser = argparse.ArgumentParser(description='Unet Predict Info')
    parser.add_argument('--image_path', default='data/voc_dev/JPEGImages/T2_00048.jpg', help='需要预测的文件路径')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--input_size', default=(512, 512), help='the model input image size')
    parser.add_argument('--backbone_type', type=str, default="resnet50", help='选择主干网络')
    parser.add_argument('--num_classes', type=int, default=7, help='目标类别数，对应网络的输出特征通道数')
    parser.add_argument('--weight_path', default="weight/init_unet.pth", help='加载训练好的模型权重')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 实例化UNet网络
    unet = Unet(backbone_type=args.backbone_type, num_classes=args.num_classes)
    unet.to(device)
    if args.weight_path is not None:
        unet.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=True)
    unet.eval()

    basename = os.path.basename(args.image_path)
    image = Image.open(args.image_path)
    # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    image = cvtColor(image)
    # 对输入图像进行一个备份，后面用于绘图
    old_img = copy.deepcopy(image)
    original_h = np.array(image).shape[0]
    original_w = np.array(image).shape[1]
    # 给图像增加灰条，实现不失真的resize
    image_data, nw, nh = resize_image(image, args.input_size)
    # 进行维度扩充，即添加上batch_size维度
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        # 数据格式转换 np -> tensor -> tensor gpu/cpu
        images = torch.from_numpy(image_data).to(device)
        # 模型对输入的图片进行推理
        outputs = unet(images)
        pr = outputs[0]

        # 取出每一个像素点的种类
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        #   将灰条部分截取掉
        pr = pr[int((args.input_size[0] - nh) // 2): int((args.input_size[0] - nh) // 2 + nh),
                int((args.input_size[1] - nw) // 2): int((args.input_size[1] - nw) // 2 + nw)]
        # 进行图片的resize
        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        # 取出每一个像素点的种类
        pr = pr.argmax(axis=-1)

        # 统计分割像素信息
        statistical_segmentation_results(prs=pr, total_classes=args.num_classes, original_size=(original_w, original_h))
        # 显示分割结果
        image = show_segmentation_results(origi_img=old_img, prs=pr, total_classes=args.num_classes)

        image.save(os.path.join('output', 'pred_' + basename))

    print(args.image_path)
