import argparse
import glob
import logging
import os
import shutil

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_dir_in', default='/home/manu/tmp/runs/manu_detect/labels/')
    parser.add_argument('--imgs_dir_in', default='/home/manu/tmp/BOSH-FM数据采集-samples-merge/')
    parser.add_argument('--labels_dir_out', default='/home/manu/tmp/samples_pick/labels')
    parser.add_argument('--imgs_dir_out', default='/home/manu/tmp/samples_pick/images')
    return parser.parse_args()


def run(args):
    make_dirs(args.labels_dir_out, reset=True)
    make_dirs(args.imgs_dir_out, reset=True)
    label_files = glob.glob(os.path.join(args.labels_dir_in, '*.txt'))
    for label_file in label_files:
        file_name = os.path.splitext(os.path.basename(label_file))[0]
        img_file = file_name + '.jpg'
        img_path = os.path.join(args.imgs_dir_in, img_file)
        if os.path.exists(img_path):
            shutil.copy2(label_file, os.path.join(args.labels_dir_out, os.path.basename(label_file)))
            shutil.copy2(img_path, os.path.join(args.imgs_dir_out, img_file))
        else:
            logging.warning(f"Image file {img_file} does not exist, skipping.")


def main():
    set_logging()
    args = parse_args()
    logging.info(f"Arguments: {args}")
    run(args)
    logging.info("Operation completed.")


if __name__ == '__main__':
    main()
