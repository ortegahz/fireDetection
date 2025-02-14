import os
import shutil

from tqdm import tqdm

from utils import make_dirs

images_folders = [
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v0/mm_results/images/pseudof',
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v1/jiu-shiwai-pic-merge-pick-filtered/images',
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v1/pics_pick_filtered/images',
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v1/yolo11_bgs_20241115',
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v1/yolo11_bgs_20241118',
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v1/yolo11_bgs_20241120',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/d-fire-jb/images',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/fish-jb/images',  # new
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/others/算法图片_labelme/images',
    '/media/manu/ST2000DM005-2U91/fire/data/others/20241127_pos_pics_merge_labelme/images/',
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v1/yolo11_bgs_20241209',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/aigc_20241230/v01merge_jb/images',  # new
]
labels_folders = [
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v0/mm_results/labels/pseudof',
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v1/jiu-shiwai-pic-merge-pick-filtered/labels',
    '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v1/pics_pick_filtered/labels',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/d-fire-jb/labels',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/fish-jb/labels',  # new
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/others/算法图片_labelme/labels_txt',
    '/media/manu/ST2000DM005-2U91/fire/data/others/20241127_pos_pics_merge_labelme/labels/',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/aigc_20241230/v01merge_jb/labels',  # new
]
output_folder = '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/out_all'
name_subset = 'pseudof'

make_dirs(output_folder, reset=True)
images_output = os.path.join(output_folder, 'images', name_subset)
labels_output = os.path.join(output_folder, 'labels', name_subset)
make_dirs(images_output, reset=True)
make_dirs(labels_output, reset=True)


def copy_files_with_counter(src_folder, dst_folder):
    file_list = os.listdir(src_folder)
    for file in tqdm(file_list, desc=f"Copying from {src_folder}"):
        src_file_path = os.path.join(src_folder, file)
        if os.path.isfile(src_file_path):
            base, ext = os.path.splitext(file)
            dst_file_name = f"{base}{ext}"
            dst_file_path = os.path.join(dst_folder, dst_file_name)
            shutil.copy(src_file_path, dst_file_path)

# TODO： consider same name files overwrite

for images_folder in images_folders:
    if os.path.exists(images_folder):
        copy_files_with_counter(images_folder, images_output)

for labels_folder in labels_folders:
    if os.path.exists(labels_folder):
        copy_files_with_counter(labels_folder, labels_output)

print('All files have been merged into the output folder.')
