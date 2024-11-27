import cv2
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

def jpg_to_rgb(pic_path, to_path,width=1440, height=1080):
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
        rgb_name = dir_name.split("_")[0] + "00000" + dir_name.split("_")[-1]
        # print(rgb_name)
        with open(os.path.join(to_path , rgb_name + ".rgb"), "wb") as f:
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

if __name__ == '__main__':
    rgb_path = '/home/manu/tmp/shiyanshi(old)'
    output_path = '/home/manu/tmp/2'

    os.makedirs(output_path, exist_ok=True)
    rgb_to_jpg(rgb_path, output_path)




# if __name__ == '__main__':
    # # pic_path = "E:\\video\pictures"
    # pic_path = r"E:\Desktop\tmp\images\smoke_20240530\vlc-record-2024-05-29-12h36m14s-rtsp___172.20.20.124_visi_stream-"
    # # to_path = "E:\\video\pictures\\"
    # to_path = r"E:\Desktop\tmp\images\smoke_20240530_\vlc-record-2024-05-29-12h36m14s-rtsp___172.20.20.124_visi_stream-"
    
    # os.makedirs(to_path,exist_ok=True)
    # jpg_to_rgb(pic_path, to_path)
