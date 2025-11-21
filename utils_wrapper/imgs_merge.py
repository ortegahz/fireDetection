import argparse
import glob
import logging
import os

import cv2
from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_in',
                        default='/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/未打标签/')
    parser.add_argument('--dir_root_out',
                        default='/home/manu/mnt/ST2000DM005-2U91/fire/data/D568.火焰，烟雾数据集/imgs_unlabeled_merge')
    return parser.parse_args()


def run(args):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    image_paths = []
    root_imgs = args.dir_root_in
    output_dir = args.dir_root_out
    make_dirs(output_dir, reset=True)

    all_image_paths = []
    for ext in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(root_imgs, '**', ext), recursive=True))

    cnt = 0
    for image_path in tqdm(all_image_paths, desc="Processing images"):
        relative_path = os.path.relpath(image_path, root_imgs)
        # Generate a new image name with a counter
        new_image_name = '_'.join(relative_path.split(os.sep))
        new_image_name_jpg = f'{cnt}_{new_image_name}.jpg'
        new_image_path = os.path.join(output_dir, new_image_name_jpg)

        # Ensure the resulting path length is within limits
        if len(new_image_path) > 255:
            logging.warning(f"File path too long, skipping: {new_image_path}")
            continue

        # Read image using OpenCV
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f"Failed to load image at path: {image_path}")
                continue

            # Save image as JPG using OpenCV
            cv2.imwrite(new_image_path, img)
            image_paths.append(new_image_path)
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")

        cnt += 1


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
