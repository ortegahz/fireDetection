import argparse
import glob
import logging
import os

from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dir_root_in',
    #                     default='/home/Huangzhe/test/FireDetectionV9_20241012_4classes')
    # parser.add_argument('--dir_root_out',
    #                     default='/home/Huangzhe/test/FireDetectionV9_20241012_4classes')
    parser.add_argument('--dir_root_in', default='/home/Huangzhe/test/person_sorted/')
    parser.add_argument('--dir_root_out', default='/home/Huangzhe/test/person_sorted/')
    parser.add_argument('--subset', default='train')  # val or train
    return parser.parse_args()


def run(args):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    image_paths = []
    root_imgs = os.path.join(args.dir_root_in, 'images')
    image_dir = os.path.join(root_imgs, args.subset)
    output_txt_file = os.path.join(args.dir_root_in, args.subset + '.txt')
    for ext in image_extensions:
        for image_path in glob.glob(os.path.join(image_dir, '**', ext), recursive=True):
            image_path_out = image_path.replace(args.dir_root_in, args.dir_root_out)
            image_paths.append(image_path_out)
    with open(output_txt_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
