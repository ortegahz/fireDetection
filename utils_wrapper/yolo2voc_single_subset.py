import argparse
import glob
import json
import logging
import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_dir_in', default='/home/manu/mnt/ST8000DM004-2U91/jb_raw/03数据标注-samples-merge-pick-smoke/labels')
    parser.add_argument('--imgs_dir_in', default='/home/manu/mnt/ST8000DM004-2U91/jb_raw/03数据标注-samples-merge-pick-smoke/images')
    parser.add_argument('--output_dir', default='/home/manu/tmp/smoke_unlabeled/VOCdevkit/VOC2007')
    parser.add_argument('--class_mapping', default='{"0": "smoke", "1": "wire", "2": "dense"}')
    return parser.parse_args()


def create_voc_xml(output_path, image_path, image_width, image_height, yolo_labels, class_mapping):
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = "VOC2007"

    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "The VOC2007 Database"
    annotation_source = ET.SubElement(source, "annotation")
    annotation_source.text = "PASCAL VOC2007"
    image = ET.SubElement(source, "image")
    image.text = "flickr"
    flickrid = ET.SubElement(source, "flickrid")
    flickrid.text = "194179466"

    owner = ET.SubElement(annotation, "owner")
    flickrid = ET.SubElement(owner, "flickrid")
    flickrid.text = "monsieurrompu"
    name = ET.SubElement(owner, "name")
    name.text = "Thom Zemanek"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_width)
    height = ET.SubElement(size, "height")
    height.text = str(image_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for label in yolo_labels:
        if len(label.split()) == 6:
            cls, center_x, center_y, box_w, box_h, conf = map(float, label.split())
        else:
            cls, center_x, center_y, box_w, box_h = map(float, label.split())

        # Convert normalized coordinates to absolute coordinates
        center_x_abs = center_x * image_width
        center_y_abs = center_y * image_height
        box_w_abs = box_w * image_width
        box_h_abs = box_h * image_height

        xmin = int(center_x_abs - (box_w_abs / 2))
        ymin = int(center_y_abs - (box_h_abs / 2))
        xmax = int(center_x_abs + (box_w_abs / 2))
        ymax = int(center_y_abs + (box_h_abs / 2))

        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_mapping[str(int(cls))]
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "1"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin_elem = ET.SubElement(bndbox, "xmin")
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox, "ymin")
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox, "xmax")
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox, "ymax")
        ymax_elem.text = str(ymax)

    # Write to XML file
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml_str)


def find_image_file(imgs_dir_in, subset, file_name):
    """
    Find the image file with the given file name (without extension) in the specified directory and subset.
    """
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    # extensions = ['jpg']
    for ext in extensions:
        img_file = f"{file_name}.{ext}"
        img_path = os.path.join(imgs_dir_in, subset, img_file)
        if os.path.exists(img_path):
            return img_path
    return None


def convert_yolo_to_voc(labels_dir_in, imgs_dir_in, output_dir, class_mapping):
    annotations_dir = os.path.join(output_dir, "Annotations")
    jpegimages_dir = os.path.join(output_dir, "JPEGImages")
    imagesets_dir = os.path.join(output_dir, "ImageSets", "Main")
    make_dirs(annotations_dir)
    make_dirs(jpegimages_dir)
    make_dirs(imagesets_dir)

    # subsets = os.listdir(labels_dir_in)
    subsets = ['train']
    cnt = 0
    for subset in subsets:
        label_files = glob.glob(os.path.join(labels_dir_in, '*.txt'))
        file_list = []
        for label_file in tqdm(label_files, desc=f"Processing labels in {subset}"):
            file_name = os.path.splitext(os.path.basename(label_file))[0]
            img_path = find_image_file(imgs_dir_in, '', file_name)
            if img_path:
                # Read image size
                image = cv2.imread(img_path)
                image_height, image_width, _ = image.shape

                # Read YOLO label
                with open(label_file, 'r') as f:
                    yolo_labels = f.readlines()

                # Create VOC XML
                xml_output_path = os.path.join(annotations_dir, f'{file_name}.xml')
                create_voc_xml(xml_output_path, img_path, image_width, image_height, yolo_labels, class_mapping)

                # Copy image to JPEGImages directory
                # img_output_path = os.path.join(jpegimages_dir, f'{os.path.basename(img_path)}')
                # shutil.copyfile(img_path, img_output_path)
                cnt += 1
                # Add to file list
                file_list.append(f'{file_name}')
            else:
                logging.warning(f"Image file for {file_name} does not exist in subset {subset}, skipping.")

        # Write file list for the current subset
        with open(os.path.join(imagesets_dir, f"{subset}.txt"), "w") as f:
            f.write("\n".join(file_list) + "\n")
    logging.info(f'cnt --> {cnt}')


def main():
    set_logging()
    args = parse_args()
    logging.info(f"Arguments: {args}")

    make_dirs(args.output_dir, reset=True)

    # Parse class mapping from JSON string
    class_mapping = json.loads(args.class_mapping)
    convert_yolo_to_voc(args.labels_dir_in, args.imgs_dir_in, args.output_dir, class_mapping)

    logging.info("Operation completed.")


if __name__ == '__main__':
    main()
