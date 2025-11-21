import os
import shutil

from tqdm import tqdm

from utils import make_dirs

# Define source image and label folders
images_folders = [
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/01.fire 118 detection/JPEGImages',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/02.fire 487 detection/JPEGImages',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/03.fire_2059_detection/JPEGImages',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/04. 474 smoke＋fire/JPEGImages',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/05. 100 Fire and Smoke Dataset/Datacluster Fire and Smoke Sample/Datacluster Fire and Smoke Sample/',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/06.39000/',
]

labels_folders = [
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/01.fire 118 detection/Annotations',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/02.fire 487 detection/JPEGImages/Annotations',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/03.fire_2059_detection/Annotations',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/04. 474 smoke＋fire/Annotations',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/05. 100 Fire and Smoke Dataset/Annotations/Annotations/',
    '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/06.39000/',
]

# Output directories
output_folder = '/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/打好标签/out_all'
name_subset = 'D568'
images_output = os.path.join(output_folder, 'images', name_subset)
labels_output = os.path.join(output_folder, 'labels', name_subset)

# Create output directories
make_dirs(images_output, reset=True)
make_dirs(labels_output, reset=True)


def copy_files_with_counter(images_src_folder, labels_src_folder, img_exts, start_cnt=0):
    """Copies images and their corresponding label files to the destination, ensuring names match."""
    image_files = []
    for ext in img_exts:
        image_files.extend([f for f in os.listdir(images_src_folder) if f.endswith(ext)])

    cnt = start_cnt
    for img_file in tqdm(image_files, desc=f"Copying from {images_src_folder}"):
        base_name = os.path.splitext(img_file)[0]

        # Find corresponding label file
        label_file = base_name + '.xml'
        label_src_path = os.path.join(labels_src_folder, label_file)
        img_src_path = os.path.join(images_src_folder, img_file)

        if os.path.isfile(label_src_path) and os.path.isfile(img_src_path):
            # Prefix files with a counter
            dst_img_file_name = f"{cnt}_{base_name}{os.path.splitext(img_file)[1]}"
            dst_label_file_name = f"{cnt}_{base_name}.xml"
            dst_img_path = os.path.join(images_output, dst_img_file_name)
            dst_label_path = os.path.join(labels_output, dst_label_file_name)

            shutil.copy(img_src_path, dst_img_path)
            shutil.copy(label_src_path, dst_label_path)
            cnt += 1


def merge_files():
    img_exts = ['.jpg', '.jpeg', '.png']

    # Process each pair of image and label folders
    for img_folder, label_folder in zip(images_folders, labels_folders):
        if os.path.exists(img_folder) and os.path.exists(label_folder):
            copy_files_with_counter(img_folder, label_folder, img_exts)

    print('All files have been merged into the output folder.')


if __name__ == "__main__":
    merge_files()
