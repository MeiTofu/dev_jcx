import copy
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F

from models.unet import Unet
from utils.util import CLASSES as name_classes


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

def preprocess_input(image):
    image /= 255.0
    return image


def draw_filled_rectangle(draw, rectangle_position=(50, 50), rectangle_size=100, rectangle_color=(255, 0, 0)):
    """
        输入 ImageDraw 对象绘制一个填充指定颜色的矩形并返回结果。
    :param draw: ImageDraw.Draw(image)
    :param rectangle_position: 矩形位置
    :param rectangle_size: 矩形大小
    :param rectangle_color: 矩形填充颜色
    :return: ImageDraw
    """

    rectangle_coordinates = [
        rectangle_position,
        (rectangle_position[0] + rectangle_size, rectangle_position[1]),
        (rectangle_position[0] + rectangle_size, rectangle_position[1] + rectangle_size//2),
        (rectangle_position[0], rectangle_position[1] + rectangle_size//2),
    ]

    draw.polygon(rectangle_coordinates, outline=None, fill=rectangle_color)
    return draw


def add_info_to_image(image, scale_factor=20, font_color=(0, 0, 0), colors=None, name_classes=None):
    """
        向图像中添加文本信息饼返回处理好的图像
    :param image: Image
    :param scale_factor: 文本相对于图像的缩放因子
    :param font_color: 文本颜色
    :param colors: 矩形填充颜色
    :param name_classes: 类别名称
    :return: Image
    """
    if colors is None:
        colors = [(255, 69, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
    if name_classes is None:
        name_classes = ["H", "T1", "T2", "T3", "T4", "T5"]
    w, h = image.size
    unit = w // scale_factor
    distance = (w - unit * 2) // len(name_classes)
    # 自适应字体大小
    font_size = unit
    # 字体位置：底部
    x, y = (unit, h - font_size)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("utils/font/arial.ttf", size=font_size)
    # # 载入默认字体
    # default_font = ImageFont.load_default()
    # # 获取默认字体的名称
    # default_font_name = default_font.getname()[0]
    # font = ImageFont.truetype(default_font_name, font_size)

    for i in range(len(name_classes)):
        text = name_classes[i]
        position = (x, y - unit)
        position_text = (x, y)
        rectangle_size = unit
        draw = draw_filled_rectangle(draw, position, rectangle_size, rectangle_color=colors[i])
        draw.text(position_text, text, font=font, fill=font_color)
        x = x + distance

    return image



if __name__ == "__main__":
    print("run unet")
    model_path = "run/20240324-12_06/last_loss_4.2435.pth"
    num_classes = 7
    input_shape = [512, 512]
    # name_classes = ["background", "H", "T1", "T2", "T3", "T4", "T5"]
    colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]

    # 实例化UNet网络
    unet = Unet(backbone_type="resnet50", num_classes=7, pretrained=False)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    unet.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    unet.eval()

    img = 'data/dev_diabetes/JPEGImages/H_00086.jpg'
    basename = os.path.basename(img)
    image = Image.open(img)
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)
    # ---------------------------------------------------#
    #   对输入图像进行一个备份，后面用于绘图
    # ---------------------------------------------------#
    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, input_shape)
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        # if cuda:
        #     images = images.cuda()

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        pr = unet(images)[0]
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#
        pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh), \
             int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
        # ---------------------------------------------------#
        #   进行图片的resize
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = pr.argmax(axis=-1)

        classes_nums = np.zeros([num_classes])
        total_points_num = orininal_h * orininal_w
        print('-' * 63)
        print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
        print('-' * 63)
        for i in range(num_classes):
            num = np.sum(pr == i)
            ratio = num / total_points_num * 100
            if num > 0:
                print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                print('-' * 63)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)

        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * colors[c][0]).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * colors[c][1]).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * colors[c][2]).astype('uint8')

        my_image = np.array(old_img)
        my_image[:, :, :][seg_img[:, :, :] > 0] = 255
        image = Image.fromarray(np.uint8(my_image))

        image = add_info_to_image(image)

        image.save("show.jpg")

    print(img)

