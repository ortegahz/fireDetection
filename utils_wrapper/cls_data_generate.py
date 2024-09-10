import argparse
import glob
import logging
import os
import random

import cv2
from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_dir_in', default='/home/manu/mnt/8gpu_3090/test/runs/test/manu_detect10/labels')
    parser.add_argument('--imgs_dir_in', default='/home/manu/mnt/8gpu_3090/test/coco/coco/train2017')
    # parser.add_argument('--labels_dir_in', default='/home/manu/tmp/mm_results/labels/pseudo/')
    # parser.add_argument('--imgs_dir_in', default='/home/manu/tmp/mm_results/images/pseudo/')
    parser.add_argument('--output_dir', default='/home/manu/tmp/labels_show_results')
    parser.add_argument('--patch_size', type=int, default=50,
                        help='Percentage to expand the bounding box')  # Default to 50%
    parser.add_argument('--sample_size', type=int, default=18061,
                        help='Number of samples to process')
    return parser.parse_args()


def expand_bbox(x, y, w, h, img_width, img_height, percentage=50):
    # Expand the bounding box by a certain percentage
    dw = w * percentage / 100
    dh = h * percentage / 100
    x -= dw / 2
    y -= dh / 2
    w += dw
    h += dh

    # Ensure the bounding box is within image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    return int(x), int(y), int(w), int(h)


def draw_boxes_on_image(img_path, label_path, patch_size):
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    with open(label_path, 'r') as f:
        labels = f.readlines()

    patches = []

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

        # Expand bounding box
        top_left_x, top_left_y, box_w, box_h = expand_bbox(top_left_x, top_left_y, box_w, box_h, width, height,
                                                           patch_size)
        bottom_right_x = top_left_x + box_w
        bottom_right_y = top_left_y + box_h

        # Crop the expanded bounding box as a patch
        patch = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        patches.append((patch, 'positive'))

    return img, patches


def run(args):
    make_dirs(args.output_dir, reset=True)
    pos_patch_dir = os.path.join(args.output_dir, 'raw_patches')
    make_dirs(pos_patch_dir, reset=True)

    label_files = glob.glob(os.path.join(args.labels_dir_in, '*.txt'))

    logging.info(f'len(label_files) --> {len(label_files)}')
    # if len(label_files) > args.sample_size:
    #     label_files = random.sample(label_files, args.sample_size)

    for i, label_file in enumerate(tqdm(label_files, desc="Processing labels")):
        file_name = os.path.splitext(os.path.basename(label_file))[0]
        img_file = file_name + '.jpg'
        img_path = os.path.join(args.imgs_dir_in, img_file)

        if os.path.exists(img_path):
            _, patches = draw_boxes_on_image(img_path, label_file, args.patch_size)

            # Save patches
            for j, (patch, label) in enumerate(patches):
                if patch is not None and patch.size > 0:
                    patch_path = os.path.join(pos_patch_dir, f'{i}_{j}_{label}.jpg')
                    cv2.imwrite(patch_path, patch)
                else:
                    logging.warning(f"Patch {i}_{j}_{label} is empty and will not be saved.")
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
