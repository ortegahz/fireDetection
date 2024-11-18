import os
import shutil
from utils import set_logging, make_dirs

# 定义文件夹路径
input_folder = '/home/manu/tmp/labels_show_results/'  # 包含jpg图片的文件夹
search_folder = '/media/manu/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/jiu-shiwai-pic-merge-pick/'  # 查找的图片和标签文件夹
output_folder = '/media/manu/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/jiu-shiwai-pic-merge-pick-filtered/'  # 输出文件夹

# 在输出文件夹中创建images和labels子文件夹
make_dirs(output_folder, reset=True)
images_output = os.path.join(output_folder, 'images')
labels_output = os.path.join(output_folder, 'labels')
make_dirs(images_output, reset=True)
make_dirs(labels_output, reset=True)

images_input = os.path.join(search_folder, 'images')
labels_input = os.path.join(search_folder, 'labels')

# 遍历输入文件夹中的文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # 去除文件扩展名（即 ".jpg"）
        basename = filename.rsplit('.', 1)[0]  # 得到 "0_J-D-10m-003_frame_000104"

        # 去掉前缀 "0_"
        name = basename.split('_', 1)[1]

        # 构造在search_folder中查找的对应图像文件路径
        corresponding_image_path = os.path.join(images_input, f'{name}.jpg')

        # 构造label文件路径
        label_file_path = os.path.join(labels_input, f'{name}.txt')

        # 检查对应的图片和标签是否存在
        if os.path.exists(corresponding_image_path) and os.path.exists(label_file_path):
            # 拷贝对应的图片到images_output
            shutil.copy(corresponding_image_path, images_output)

            # 拷贝对应的标签到labels_output
            shutil.copy(label_file_path, labels_output)
            print(f'Copied: {name}.jpg and {name}.txt')
