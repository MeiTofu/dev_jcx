#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Description:
"""
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt


def statistics_DFU_classes(src_dataset_path):
    image_list = os.listdir(src_dataset_path)
    images_count = len(image_list)
    count = 0
    label_quantity_dict = {}
    print("数据集样本数量：", images_count // 2)
    for i in range(0, images_count):
        path = os.path.join(src_dataset_path, image_list[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
            for shape in data['shapes']:
                label_name = shape['label']
                label_quantity_dict[label_name] = label_quantity_dict.get(label_name, 0) + 1
            count += 1
            print('{}/{}'.format(count, images_count // 2), 'processing ' + image_list[i].split(".")[0] + '.jpg')
    return label_quantity_dict


def draw_statistics_result(classes_label_dict, save_path=None):
    # 根据字典的值进行排序
    sorted_label_dict = dict(sorted(classes_label_dict.items(), key=lambda x: x[1], reverse=True))

    # 遍历键
    categories = [key for key in sorted_label_dict]
    # 遍历值
    values = [value for value in sorted_label_dict.values()]

    # 创建柱状图
    bars = plt.bar(categories, values)

    # 在每个柱形上显示数值
    for bar in bars:
        y_value = bar.get_height()  # 获取柱形的高度（即数值）
        plt.text(bar.get_x() + bar.get_width() / 2, y_value + 0.5, y_value, ha='center', va='bottom')

    # 添加标题和标签
    plt.title('DFU Dataset Statistics Result')
    plt.xlabel('Categories')
    plt.ylabel('Quantity')
    # 显示柱状图
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


if __name__ == '__main__':
    print("statistics_dataset")

    # 源数据集路径
    my_dataset_path = "../../dataset/todo/done_v1.1"
    label_dict = statistics_DFU_classes(my_dataset_path)
    print(label_dict)
    draw_statistics_result(label_dict)

    print("ok")
