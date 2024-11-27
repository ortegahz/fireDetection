import glob
import os
import shutil

from tqdm import tqdm  # Import tqdm for progress bar

from utils import make_dirs


def replace_class_in_file(src_file_path, dst_file_path):
    with open(src_file_path, 'r') as src_file:
        lines = src_file.readlines()

    with open(dst_file_path, 'w') as dst_file:
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0] == '0':  # smoke
                parts[0] = '3'
            elif len(parts) > 0 and parts[0] == '1':  # fire
                parts[0] = '0'
            dst_file.write(' '.join(parts) + '\n')


def process_files(image_dir, label_dir, output_dir):
    image_out_dir = os.path.join(output_dir, 'images')
    label_out_dir = os.path.join(output_dir, 'labels')

    # Ensure output directories exist
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    # Use recursive glob pattern to find images in all subdirectories
    image_extensions = ['**/*.png', '**/*.jpg', '**/*.jpeg']  # Adjust extensions as needed
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext), recursive=True))

    txt_files = glob.glob(os.path.join(label_dir, '**/*.txt'), recursive=True)

    # Process label files with progress bar
    for txt_file in tqdm(txt_files, desc="Processing label files", unit="file"):
        filename = os.path.basename(txt_file)
        output_file_path = os.path.join(label_out_dir, filename)

        replace_class_in_file(txt_file, output_file_path)

    # Process image files with progress bar
    for image_file in tqdm(image_files, desc="Copying image files", unit="file"):
        filename = os.path.basename(image_file)
        output_file_path = os.path.join(image_out_dir, filename)

        shutil.copy(image_file, output_file_path)


image_dir = '/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/d-fire'
label_dir = image_dir
output_dir = '/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/d-fire-jb'

make_dirs(output_dir, reset=True)

process_files(image_dir, label_dir, output_dir)
