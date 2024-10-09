import argparse
import glob
import logging
import os
import shutil

from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_in',
                        default='/media/manu/ST8000DM004-2U91/jb_raw/03数据标注-samples/')
    parser.add_argument('--dir_root_out',
                        default='/media/manu/ST8000DM004-2U91/jb_raw/03数据标注-samples-merge/')
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

    for image_path in tqdm(all_image_paths, desc="Processing images"):
        relative_path = os.path.relpath(image_path, root_imgs)
        new_image_name = '_'.join(relative_path.split(os.sep))
        new_image_path = os.path.join(output_dir, new_image_name)

        if len(new_image_path) > 255:
            logging.warning(f"File path too long, skipping: {new_image_path}")
            continue

        try:
            shutil.copy2(image_path, new_image_path)
            image_paths.append(new_image_path)
        except OSError as e:
            logging.error(f"Error copying file {image_path} to {new_image_path}: {e}")


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
