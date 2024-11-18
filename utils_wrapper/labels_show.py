import argparse
import glob
import logging
import os

import cv2
from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_dir_in', default='/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/out_all/labels/pseudof')
    parser.add_argument('--imgs_dir_in', default='/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/out_all/images/pseudof')
    # parser.add_argument('--labels_dir_in', default='/media/manu/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/jiu-shiwai-pic-merge-pick/labels')
    # parser.add_argument('--imgs_dir_in', default='/media/manu/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/jiu-shiwai-pic-merge-pick/images')
    parser.add_argument('--output_dir', default='/home/manu/tmp/labels_show_results')
    return parser.parse_args()


def draw_boxes_on_image(img_path, label_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        if len(label.split()) == 6:
            cls, center_x, center_y, box_w, box_h, conf = map(float, label.split())
        else:
            cls, center_x, center_y, box_w, box_h = map(float, label.split())
        center_x *= width
        center_y *= height
        box_w *= width
        box_h *= height

        top_left_x = int(center_x - box_w / 2)
        top_left_y = int(center_y - box_h / 2)
        bottom_right_x = int(center_x + box_w / 2)
        bottom_right_y = int(center_y + box_h / 2)

        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        cv2.putText(img, str(int(cls)), (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return img


def run(args):
    make_dirs(args.output_dir, reset=True)
    label_files = glob.glob(os.path.join(args.labels_dir_in, '*.txt'))

    for i, label_file in enumerate(tqdm(label_files, desc="Processing labels")):
        file_name = os.path.splitext(os.path.basename(label_file))[0]
        img_file = file_name + '.jpg'
        img_path = os.path.join(args.imgs_dir_in, img_file)
        output_path = os.path.join(args.output_dir, f'{i}_{img_file}')

        if os.path.exists(img_path):
            img_with_boxes = draw_boxes_on_image(img_path, label_file)
            cv2.imwrite(output_path, img_with_boxes)
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
