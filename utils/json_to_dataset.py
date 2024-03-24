#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 20:03
@Message: null
"""
import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

from utils.util import CLASSES, generate_dir


def json2dataset(src_dataset_path, images_save_path, masks_save_path):
    """
    将包含原图和 json 标注文件的数据转换为图像和对应 mask 格式
    :param src_dataset_path: 原始数据集路径
    :param images_save_path: 目标图像文件路径
    :param masks_save_path: 目标 mask 文件路径
    :return:
    """
    image_list = os.listdir(src_dataset_path)
    images_count = len(image_list)
    count = 0
    print("数据集样本数量：", images_count // 2)
    for i in range(0, images_count):
        path = os.path.join(src_dataset_path, image_list[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            PIL.Image.fromarray(img).save(osp.join(images_save_path, image_list[i].split(".")[0] + '.jpg'))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = CLASSES.index(name)
                new = new + index_all * (np.array(lbl) == index_json)

            utils.lblsave(osp.join(masks_save_path, image_list[i].split(".")[0] + '.png'), new)
            count += 1
            print('{}/{}'.format(count, images_count//2), 'Saved ' + image_list[i].split(".")[0] + '.jpg and ' + image_list[i].split(".")[0] + '.png')


if __name__ == '__main__':
    print("json_to_dataset")

    jpgs_save_path = "../data/dev_datasets/JPEGImages"
    pngs_save_path = "../data/dev_datasets/SegmentationClass"
    my_dataset_path = "../data/my_dev_dataset"

    # 若目标文件夹不存在，则自动创建
    generate_dir(jpgs_save_path)
    generate_dir(pngs_save_path)

    json2dataset(my_dataset_path, jpgs_save_path, pngs_save_path)

    print("success")
