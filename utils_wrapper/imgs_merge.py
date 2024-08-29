import argparse
import glob
import logging
import os
import shutil

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_in',
                        default='/home/manu/tmp/BOSH-FM数据采集-samples/')
    parser.add_argument('--dir_root_out',
                        default='/home/manu/tmp/BOSH-FM数据采集-samples-merge/')
    return parser.parse_args()


def run(args):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    image_paths = []
    root_imgs = args.dir_root_in
    output_dir = args.dir_root_out
    make_dirs(output_dir, reset=True)

    for ext in image_extensions:
        for image_path in glob.glob(os.path.join(root_imgs, '**', ext), recursive=True):
            relative_path = os.path.relpath(image_path, root_imgs)
            new_image_name = '_'.join(relative_path.split(os.sep))
            new_image_path = os.path.join(output_dir, new_image_name)
            shutil.copy2(image_path, new_image_path)
            image_paths.append(new_image_path)


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
