import argparse
import glob
import logging
import os
import shutil

import cv2
import matplotlib.pyplot as plt

from utils import set_logging


def make_dirs(path, reset=False):
    if reset and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Process YOLO annotations and visualize HSV patches.")
    parser.add_argument('--img_dir', default='/home/manu/tmp/samples_pick/images')
    parser.add_argument('--anno_dir', default='/home/manu/tmp/samples_pick/labels')
    return parser.parse_args()


def read_yolo_annotations(file_path):
    with open(file_path, 'r') as file:
        annotations = file.readlines()
    return [line.strip().split() for line in annotations]


def get_img_patch(img, bbox, img_width, img_height):
    label, cx, cy, w, h = map(float, bbox)
    cx, cy, w, h = cx * img_width, cy * img_height, w * img_width, h * img_height
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def plot_hsv_space(hsv_patch):
    h, s, v = cv2.split(hsv_patch)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(h.flatten(), s.flatten(), v.flatten(),
               c=cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2RGB).reshape(-1, 3) / 255.0, marker='o')
    ax.set_xlabel('Hue (0-179)')
    ax.set_ylabel('Saturation (0-255)')
    ax.set_zlabel('Value (0-255)')
    ax.set_xlim(0, 179)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    plt.show()


def process_image(img_path, anno_path):
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    annotations = read_yolo_annotations(anno_path)
    for annotation in annotations:
        img_patch, (x1, y1, x2, y2) = get_img_patch(img, annotation, img_width, img_height)

        # Draw rectangle on the original image
        img_with_bbox = img.copy()
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the original image with bounding box
        plt.figure()
        plt.imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
        plt.title('Original Image with Bounding Box')
        plt.axis('off')
        plt.show()

        img_patch_hsv = cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)
        plot_hsv_space(img_patch_hsv)


def run(args):
    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    for ext in img_extensions:
        for img_path in glob.glob(os.path.join(args.img_dir, ext)):
            base_name = os.path.basename(img_path)
            anno_path = os.path.join(args.anno_dir, os.path.splitext(base_name)[0] + '.txt')
            if os.path.exists(anno_path):
                logging.info(f'Processing {img_path} with {anno_path}')
                process_image(img_path, anno_path)
            else:
                logging.warning(f'Annotation file not found for {img_path}')


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
