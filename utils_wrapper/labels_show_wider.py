import argparse
import glob
import logging
import os

import cv2
from tqdm import tqdm

# 假设 utils 文件在同级目录下，如果不存在请自行替换这两个函数的实现
from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    # 修改默认路径以匹配 WiderPerson 数据集结构
    parser.add_argument('--imgs_dir_in', default='/home/manu/tmp/WiderPerson/Images')
    parser.add_argument('--labels_dir_in', default='/home/manu/tmp/WiderPerson/Annotations')
    parser.add_argument('--output_dir', default='/home/manu/tmp/labels_show_results')
    parser.add_argument('--img_ext', default='jpg')
    return parser.parse_args()


def draw_boxes_on_image(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        logging.warning(f"Failed to read image: {img_path}")
        return None

    # WiderPerson 类别定义
    class_map = {
        1: "Pedestrians",
        2: "Riders",
        3: "Partially-visible",
        4: "Ignore",
        5: "Crowd"
    }

    with open(label_path, 'r') as f:
        labels = f.readlines()

    # WiderPerson 格式说明:
    # 第一行是数量，从第二行开始是标注
    # 格式: class_label x1 y1 x2 y2 (绝对坐标)

    # 跳过第一行 (labels[1:])
    for label in labels[1:]:
        parts = list(map(float, label.split()))

        # 确保数据长度足够
        if len(parts) < 5:
            continue

        cls = int(parts[0])
        x1 = int(parts[1])
        y1 = int(parts[2])
        x2 = int(parts[3])
        y2 = int(parts[4])

        # 绘制矩形框
        # 颜色使用绿色 (B, G, R) -> (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制类别标签
        label_text = f"{class_map.get(cls, str(cls))}"
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    return img


def run(args):
    make_dirs(args.output_dir, reset=True)
    label_files = glob.glob(os.path.join(args.labels_dir_in, '*.txt'))

    for i, label_file in enumerate(tqdm(label_files, desc="Processing labels")):
        # WiderPerson 的标注文件通常是 "图片名.jpg.txt"
        # 例如: 000001.jpg.txt -> 图片名应该是 000001.jpg
        # 所以这里直接去掉末尾的 .txt 即可
        img_file = os.path.basename(label_file).replace('.txt', '')

        img_path = os.path.join(args.imgs_dir_in, img_file)
        output_path = os.path.join(args.output_dir, f'{i}_{img_file}')

        if os.path.exists(img_path):
            img_with_boxes = draw_boxes_on_image(img_path, label_file)
            if img_with_boxes is not None:
                cv2.imwrite(output_path, img_with_boxes)
        else:
            # 尝试另一种情况，如果标注文件只是 ID.txt (例如 000001.txt)
            # 那么就需要拼接后缀
            file_name_no_ext = os.path.splitext(os.path.basename(label_file))[0]
            img_file_alt = file_name_no_ext + '.' + args.img_ext
            img_path_alt = os.path.join(args.imgs_dir_in, img_file_alt)

            if os.path.exists(img_path_alt):
                img_with_boxes = draw_boxes_on_image(img_path_alt, label_file)
                if img_with_boxes is not None:
                    cv2.imwrite(os.path.join(args.output_dir, f'{i}_{img_file_alt}'), img_with_boxes)
            else:
                logging.warning(f"Image file not found for label {label_file}, skipping.")


def main():
    set_logging()
    args = parse_args()
    logging.info(f"Arguments: {args}")
    run(args)
    logging.info("Operation completed.")


if __name__ == '__main__':
    main()
