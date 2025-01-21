from glob import glob
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import make_dirs


def jpg_to_rgb(pic_path, to_path, width=1440, height=1080):
    for item in tqdm(os.listdir(pic_path)):
        arr = item.strip().split('*')
        img_name = arr[0]
        picture_path = os.path.join(pic_path, img_name)
        pic_org = Image.open(picture_path)
        pic_org.resize((width, height), Image.ANTIALIAS).save(picture_path)
        img_BGR = cv2.imread(picture_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        a = img_RGB.tobytes()
        dir_name = img_name.split(".")[0]
        # print(dir_name)
        # rgb_name = dir_name.split("_")[0] + "_" + dir_name.split("_")[1] + "_" + "00000" + dir_name.split("_")[2]
        # rgb_name = dir_name.split("-")[1] + "_" + "00000" + dir_name.split("_")[1]
        # rgb_name = dir_name.split("_")[0] + "00000" + dir_name.split("_")[-1]
        rgb_name = img_name.replace('.png', '.rgb')
        # print(rgb_name)
        with open(os.path.join(to_path, rgb_name), "wb") as f:
            f.write(a)


def rgb_to_jpg(rgb_path, output_path, width=1440, height=1080):
    for item in tqdm(os.listdir(rgb_path)):
        if item.endswith(".rgb"):
            rgb_file = os.path.join(rgb_path, item)
            # 读取 RGB 数据
            with open(rgb_file, "rb") as f:
                rgb_data = f.read()
            # 将字节数据转换为 NumPy 数组
            img_array = np.frombuffer(rgb_data, dtype=np.uint8)
            # 重塑 NumPy 数组为图像形状
            img_array = img_array.reshape((height, width, 3))
            # 将 RGB 转换为 BGR 用于保存为 JPG
            img_BGR = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # 生成输出 JPG 文件名
            jpg_name = item.replace(".rgb", ".jpg")
            jpg_file = os.path.join(output_path, jpg_name)
            # 保存为 JPG 格式
            cv2.imwrite(jpg_file, img_BGR)


def rgb_to_jpg_plus(rgb_path, output_path, width=1440, height=1080):
    # 递归查找所有子文件夹中的 .rgb 文件
    rgb_files = glob(os.path.join(rgb_path, '**', '*.rgb'), recursive=True)

    for rgb_file in tqdm(rgb_files):
        # 读取 RGB 数据
        with open(rgb_file, "rb") as f:
            rgb_data = f.read()
        # 将字节数据转换为 NumPy 数组
        img_array = np.frombuffer(rgb_data, dtype=np.uint8)
        # 重塑 NumPy 数组为图像形状
        img_array = img_array.reshape((height, width, 3))
        # 将 RGB 转换为 BGR 用于保存为 JPG
        img_BGR = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 生成输出 JPG 文件相对于根文件夹的路径
        relative_path = os.path.relpath(os.path.dirname(rgb_file), rgb_path)
        output_dir = os.path.join(output_path, relative_path)
        # 创建输出目录，如果不存在
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出 JPG 文件名
        item = os.path.basename(rgb_file)
        jpg_name = item.replace(".rgb", ".jpg")
        jpg_file = os.path.join(output_dir, jpg_name)
        # 保存为 JPG 格式
        cv2.imwrite(jpg_file, img_BGR)


if __name__ == '__main__':
    # rgb_path = '/run/user/1000/gvfs/smb-share:server=172.20.254.27,share=青鸟消防智慧可视化02/00部门共享/【临时文件交换目录】/【to】汤香渝/热成像数据_20241028'
    # output_path = '/home/manu/tmp/rgb2jpg'
    #
    # make_dirs(output_path, reset=True)
    # rgb_to_jpg_plus(rgb_path, output_path)

    input_path = '/home/manu/tmp/smoke1225_8'
    output_path = '/home/manu/tmp/rgb'

    make_dirs(output_path, reset=True)
    jpg_to_rgb(input_path, output_path)

# if __name__ == '__main__':
# # pic_path = "E:\\video\pictures"
# pic_path = r"E:\Desktop\tmp\images\smoke_20240530\vlc-record-2024-05-29-12h36m14s-rtsp___172.20.20.124_visi_stream-"
# # to_path = "E:\\video\pictures\\"
# to_path = r"E:\Desktop\tmp\images\smoke_20240530_\vlc-record-2024-05-29-12h36m14s-rtsp___172.20.20.124_visi_stream-"

# os.makedirs(to_path,exist_ok=True)
# jpg_to_rgb(pic_path, to_path)
