import os
import shutil

import cv2
import numpy as np

from utils_wrapper.utils import make_dirs


def calculate_pixel_sum_normalized(image_folder, output_folder):
    total_sum_normalized = 0
    image_count = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        if 'combined' not in filename:
            continue
        file_path = os.path.join(image_folder, filename)
        # 检查文件是否是图片
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            # 使用OpenCV读取图片
            img = cv2.imread(file_path)
            if img is not None:
                # 计算图像的面积
                height, width, _ = img.shape
                area = height * width
                # 计算所有通道的像素和
                pixel_sum = np.sum(img)
                # 对像素和进行归一化
                pixel_sum_normalized = pixel_sum / area
                total_sum_normalized += pixel_sum_normalized
                image_count += 1
                print(f"Image: {filename}, Pixel Sum: {pixel_sum}, Normalized Pixel Sum: {pixel_sum_normalized:.2f}")

                # 如果归一化的像素和小于16，将图像复制到目标文件夹
                filename_rgb = filename.replace('combined', 'image')
                file_path_rgb = file_path.replace('combined', 'image')
                # if pixel_sum_normalized < 16:
                #     continue
                # shutil.copy(file_path, os.path.join(output_folder, filename))
                shutil.copy(file_path_rgb, os.path.join(output_folder, filename_rgb))
            else:
                print(f"Warning: Could not read image file {filename}")

    if image_count > 0:
        average_normalized_sum = total_sum_normalized / image_count
        print(f"Average Normalized Pixel Sum of all images: {average_normalized_sum:.2f}")
    else:
        print("No valid images found to compute.")


# 使用函数，传入图像文件夹的路径和输出文件夹的路径
# image_folder_path = '/home/manu/tmp/fire_cls_raw/fire_test_results_pos'
# output_folder_path = '/home/manu/tmp/fire_cls_raw/fire_test_results_pos_pick'
image_folder_path = '/home/manu/tmp/fire_cls_raw/fire_test_results_neg'
output_folder_path = '/home/manu/tmp/fire_cls_raw/fire_test_results_neg_pick'
make_dirs(output_folder_path, reset=True)
calculate_pixel_sum_normalized(image_folder_path, output_folder_path)
