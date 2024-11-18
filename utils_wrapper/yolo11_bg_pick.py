import glob
import os
import shutil

from tqdm import tqdm  # 从 tqdm 库中导入进度条模块

from utils import make_dirs


def copy_images_with_labels(txt_folder, jpg_folder, output_folder):
    # 搜索所有的 .txt 文件
    txt_files = glob.glob(os.path.join(txt_folder, '**', '*.txt'), recursive=True)

    # 将 .txt 文件路径以文件名为键存入字典
    txt_files_dict = {
        os.path.splitext(os.path.basename(txt_file_path))[0]: txt_file_path
        for txt_file_path in txt_files
    }

    # 搜索所有的 .jpg 文件
    jpg_files = glob.glob(os.path.join(jpg_folder, '**', '*.jpg'), recursive=True)

    # 使用 tqdm 包装 jpg_files，显示进度
    for jpg_file_path in tqdm(jpg_files, desc="Copying images", unit="file"):
        # 获取 .jpg 文件的目录和文件名部分
        dir_name = os.path.basename(os.path.dirname(jpg_file_path)).replace('_frames', '')
        file_name = os.path.splitext(os.path.basename(jpg_file_path))[0]

        # 合成用于匹配的 .txt 文件基名
        txt_base_name = f"{dir_name}_{file_name}"

        # 检查该基名是否存在于 .txt 文件的字典中
        if txt_base_name in txt_files_dict:
            # 创建新的文件名，增加原文件夹名称作为前缀
            new_file_name = f"{dir_name}_{file_name}.jpg"
            new_file_path = os.path.join(output_folder, new_file_name)

            # 复制 .jpg 文件到输出目录
            shutil.copy(jpg_file_path, new_file_path)


# 使用示例
txt_directory = '/media/manu/ST2000DM005-2U91/workspace/ultralytics/runs/detect/predict/'
jpg_directory = txt_directory
output_directory = '/home/manu/tmp/yolo11_bgs'
make_dirs(output_directory, reset=True)
copy_images_with_labels(txt_directory, jpg_directory, output_directory)
