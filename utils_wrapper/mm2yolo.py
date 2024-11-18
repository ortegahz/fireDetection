import argparse
import glob
import json
import logging
import os
import shutil

import cv2
from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_in',
                        default='/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v0/out/imgs')
    parser.add_argument('--dir_root_out',
                        default='/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v0/mm_results/')
    parser.add_argument('--dir_json',
                        default='/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/v0/out/labels')
    parser.add_argument('--score_threshold', type=float, default=0.1,
                        help='Score threshold for displaying bounding boxes')
    parser.add_argument('--subset', default='pseudof', help='Subset name')
    parser.add_argument('--force_imgs_copy', default=True)
    return parser.parse_args()


def convert_to_yolo_format(image_shape, bbox):
    img_h, img_w = image_shape[:2]
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return cx, cy, w, h


def convert_bboxes_to_yolo(image_path, json_path, score_threshold=0.3):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image = cv2.imread(image_path)
    labels = data['labels']
    scores = data['scores']
    bboxes = data['bboxes']

    cnt_valid = 0
    yolo_labels = []
    for label, score, bbox in zip(labels, scores, bboxes):
        if score > score_threshold:
            cx, cy, w, h = convert_to_yolo_format(image.shape, bbox)
            yolo_labels.append(f'{label} {cx} {cy} {w} {h}')
            cnt_valid += 1

    return yolo_labels, cnt_valid


def run(args):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    root_imgs = args.dir_root_in
    output_dir_images = os.path.join(args.dir_root_out, 'images', args.subset)
    output_dir_labels = os.path.join(args.dir_root_out, 'labels', args.subset)
    json_dir = args.dir_json

    make_dirs(args.dir_root_out, reset=True)
    make_dirs(output_dir_images, reset=True)
    make_dirs(output_dir_labels, reset=True)

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(root_imgs, '**', ext), recursive=True))

    for image_path in tqdm(image_paths, desc="Processing images"):
        relative_path = os.path.relpath(image_path, root_imgs)
        new_image_name = '_'.join(relative_path.split(os.sep))
        new_image_path = os.path.join(output_dir_images, new_image_name)
        json_path = os.path.join(json_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')
        new_label_path = os.path.join(output_dir_labels, os.path.splitext(new_image_name)[0] + '.txt')

        if args.force_imgs_copy:
            shutil.copy2(image_path, new_image_path)

        if os.path.exists(json_path):
            yolo_labels, cnt_valid = convert_bboxes_to_yolo(image_path, json_path, score_threshold=args.score_threshold)
            if cnt_valid > 0:
                shutil.copy2(image_path, new_image_path)
                with open(new_label_path, 'w') as f:
                    for label in yolo_labels:
                        f.write(label + '\n')


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
