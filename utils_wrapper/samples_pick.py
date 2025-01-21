import argparse
import glob
import logging
import os
import shutil

from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_dir_in',
                        default='/home/manu/tmp/labelme/labels_txt/',
                        help='Input directory for label files')
    parser.add_argument('--imgs_dir_in',
                        default='/home/manu/tmp/labelme/images/',
                        help='Input directory for image files')
    parser.add_argument('--labels_dir_out',
                        default='/home/manu/tmp/pics_pick/labels',
                        help='Output directory for label files')
    parser.add_argument('--imgs_dir_out',
                        default='/home/manu/tmp/pics_pick/images',
                        help='Output directory for image files')
    parser.add_argument('--conf_threshold', type=float, nargs=2, default=[0.0, 1.0],
                        help='Confidence threshold for selecting images and labels')
    parser.add_argument('--force_copy', default=True)
    return parser.parse_args()


def run(args):
    make_dirs(args.labels_dir_out, reset=True)
    make_dirs(args.imgs_dir_out, reset=True)
    label_files = glob.glob(os.path.join(args.labels_dir_in, '*.txt'))

    lower_threshold, upper_threshold = args.conf_threshold

    for label_file in tqdm(label_files, desc="Processing files", unit="file"):
        file_name = os.path.splitext(os.path.basename(label_file))[0]
        img_file = file_name + '.jpg'
        img_path = os.path.join(args.imgs_dir_in, img_file)

        if os.path.exists(img_path):
            with open(label_file, 'r') as lf:
                labels = lf.readlines()

            if args.force_copy:
                all_threshold = True
            else:
                all_threshold = all(lower_threshold <= float(label.split()[5]) <= upper_threshold for label in labels)

            if all_threshold:
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
