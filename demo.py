#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 19:32
@Message: null
"""
import torch

if __name__ == "__main__":
    print("demo")
    # raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # 生成有序数组
    arr = torch.arange(2 * 3 * 4)

    # 将数组reshape为维度为(2, 3, 4)
    arr = arr.reshape(2, 3, 4)
    print(arr)

    tp1 = torch.sum(arr, axis=[0,1])
    print(tp1)
    tp = torch.sum(arr, dim=(0, 1))
    print(tp)
