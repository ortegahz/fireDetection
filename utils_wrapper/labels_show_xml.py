import argparse
import glob
import logging
import os
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir_in',
                        default='/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/d-fire_labeled/d-fire-jb/images')
    parser.add_argument('--labels_dir_in',
                        default='/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/d-fire_labeled/d-fire-jb/labels_xml')
    parser.add_argument('--output_dir', default='/home/manu/tmp/labels_show_results')
    parser.add_argument('--img_exts', nargs='+', default=['.png', '.jpg', '.jpeg'],
                        help='List of possible image file extensions')
    return parser.parse_args()


def parse_voc_xml(label_path):
    tree = ET.parse(label_path)
    root = tree.getroot()

    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = int(float(xmlbox.find('xmin').text))
        ymin = int(float(xmlbox.find('ymin').text))
        xmax = int(float(xmlbox.find('xmax').text))
        ymax = int(float(xmlbox.find('ymax').text))
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def draw_boxes_on_image(img_path, label_path):
    img = cv2.imread(img_path)
    boxes = parse_voc_xml(label_path)

    for name, xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return img


def find_image_file(imgs_dir_in, file_name, possible_exts):
    """Try to find an image file with the base name in one of the possible extensions."""
    for ext in possible_exts:
        img_file = file_name + ext
        img_path = os.path.join(imgs_dir_in, img_file)
        if os.path.exists(img_path):
            return img_path
    return None


def run(args):
    make_dirs(args.output_dir, reset=True)
    label_files = glob.glob(os.path.join(args.labels_dir_in, '*.xml'))

    for i, label_file in enumerate(tqdm(label_files, desc="Processing labels")):
        file_name = os.path.splitext(os.path.basename(label_file))[0]
        img_path = find_image_file(args.imgs_dir_in, file_name, args.img_exts)

        if img_path:
            img_with_boxes = draw_boxes_on_image(img_path, label_file)
            output_img_file = os.path.basename(img_path)
            output_path = os.path.join(args.output_dir, f'{i}_{output_img_file}')
            cv2.imwrite(output_path, img_with_boxes)
        else:
            logging.warning(f"Image file for {file_name} with acceptable extensions does not exist, skipping.")


def main():
    set_logging()
    args = parse_args()
    logging.info(f"Arguments: {args}")
    run(args)
    logging.info("Operation completed.")


if __name__ == '__main__':
    main()
