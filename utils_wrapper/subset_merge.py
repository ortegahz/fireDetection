import argparse
import logging
import os
import shutil

from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='/home/Huangzhe/test/fire')
    parser.add_argument('--subsets', default=['v9', 'test', 'pseudof'])
    # parser.add_argument('--subsets', default=['test'])
    parser.add_argument('--output_subset', default='train', help='Name of the output merged subset')
    return parser.parse_args()


def copy_files_and_write_list(list_file, imgs_dir, labels_dir, output_imgs_dir, output_labels_dir, output_list_file,
                              progress_desc):
    with open(list_file, 'r') as f, open(output_list_file, 'a') as out_f:
        lines = f.readlines()
        for line in tqdm(lines, desc=progress_desc):
            img_file = line.strip()
            img_file_name = os.path.basename(img_file)
            label_file = os.path.splitext(img_file_name)[0] + '.txt'

            # Check if the source and destination files are the same
            src_img_path = os.path.join(imgs_dir, img_file_name)
            dst_img_path = os.path.join(output_imgs_dir, img_file_name)
            src_label_path = os.path.join(labels_dir, label_file)
            dst_label_path = os.path.join(output_labels_dir, label_file)

            if src_img_path != dst_img_path:
                shutil.copy(src_img_path, dst_img_path)
            if src_label_path != dst_label_path and os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)

            # Write to output list with absolute path
            out_f.write(os.path.abspath(dst_img_path) + '\n')


def merge_files(subsets, output_subset, base_dir):
    # Output directories
    output_imgs_dir = os.path.join(base_dir, 'images', output_subset)
    output_labels_dir = os.path.join(base_dir, 'labels', output_subset)
    output_list_file = os.path.join(base_dir, f'{output_subset}.txt')

    # Create output directories if they don't exist
    make_dirs(output_imgs_dir, reset=True)
    make_dirs(output_labels_dir, reset=True)
    if os.path.exists(output_list_file):
        os.remove(output_list_file)

    for subset in subsets:
        # Define paths for current subset
        imgs_dir = os.path.join(base_dir, 'images', subset)
        labels_dir = os.path.join(base_dir, 'labels', subset)
        list_file = os.path.join(base_dir, f'{subset}.txt')

        # Copy files from current subset
        copy_files_and_write_list(
            list_file, imgs_dir, labels_dir,
            output_imgs_dir, output_labels_dir,
            output_list_file, progress_desc=f"Processing {subset}"
        )


def main():
    set_logging()
    args = parse_args()
    logging.info(f"Arguments: {args}")
    merge_files(args.subsets, args.output_subset, args.base_dir)
    logging.info("Operation completed.")


if __name__ == '__main__':
    main()
