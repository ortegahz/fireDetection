import os
import shutil

import cv2
from tqdm import tqdm

from utils import make_dirs

imgs_folder = '/home/manu/tmp/yolo_unmerge/'
output_folder = '/home/manu/tmp/yolo_augmented_images'

subset_name = 'seq'
images_output_folder = os.path.join(output_folder, 'images', subset_name)
labels_output_folder = os.path.join(output_folder, 'labels', subset_name)

make_dirs(output_folder, reset=True)
make_dirs(images_output_folder, reset=True)
make_dirs(labels_output_folder, reset=True)

# 获取所有子目录
subdirs = [d for d in os.listdir(imgs_folder) if os.path.isdir(os.path.join(imgs_folder, d))]

# 遍历每个子文件夹，添加tqdm进度条
for subdir in tqdm(subdirs, desc="Processing subdirectories", unit="subdir"):
    subdir_path = os.path.join(imgs_folder, subdir)

    # 获取子文件夹内图片
    img_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.jpg')])

    for img_file in img_files:
        # 提取当前帧号
        base_name, _ = os.path.splitext(img_file)
        try:
            frame_number = int(base_name.split('_')[-1])
        except ValueError:
            continue

        # 找到上一帧
        prev_frame_number = frame_number - 1
        prev_img_file = img_file.replace(f"{prev_frame_number:06d}", f"{prev_frame_number:06d}")
        prev_img_path = os.path.join(subdir_path, prev_img_file)

        # 检查上一帧是否存在
        if not os.path.exists(prev_img_path):
            continue

        # 读取当前图片和上一帧图片
        img_path = os.path.join(subdir_path, img_file)
        current_img = cv2.imread(img_path)
        prev_img = cv2.imread(prev_img_path)

        # 计算灰度图差值
        gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.absdiff(gray_current, gray_prev)

        # 获取当前图片的H通道
        hsv_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2HSV)
        h_channel = hsv_current[:, :, 0]

        # 获取上一帧图片的S通道
        hsv_prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2HSV)
        s_channel = hsv_prev[:, :, 1]

        # 将通道组合成新图像
        combined_img = cv2.merge([gray_diff, h_channel, s_channel])

        # 保存新图片
        new_img_path = os.path.join(images_output_folder, img_file)
        cv2.imwrite(new_img_path, combined_img)

        # 复制对应的txt文件
        txt_file = img_file.replace('.jpg', '.txt')
        txt_src_path = os.path.join(subdir_path, txt_file)
        txt_dst_path = os.path.join(labels_output_folder, txt_file)
        if os.path.exists(txt_src_path):
            shutil.copy(txt_src_path, txt_dst_path)
