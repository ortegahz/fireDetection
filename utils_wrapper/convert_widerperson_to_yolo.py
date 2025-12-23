import argparse
import glob
import logging
import os
import shutil

import cv2
from tqdm import tqdm


# 简单的日志设置
def set_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


# 创建文件夹工具
def make_dirs(path, reset=False):
    if reset and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert WiderPerson dataset to YOLO format")

    # 输入路径配置
    parser.add_argument('--imgs_dir_in', default='/home/manu/tmp/WiderPerson/Images',
                        help='Path to WiderPerson Images folder')
    parser.add_argument('--labels_dir_in', default='/home/manu/tmp/WiderPerson/Annotations',
                        help='Path to WiderPerson Annotations folder')

    # 输出路径配置
    parser.add_argument('--output_dir', default='/home/manu/tmp/WiderPerson/widerperson_yolo',
                        help='Output root directory')

    # 转换选项
    parser.add_argument('--merge_person', action='store_true',
                        help='If True, merge class 1, 2, 3 into class 0 (person) and ignore others.')

    return parser.parse_args()


def convert_box(size, box):
    """
    将 [x1, y1, x2, y2] 转换为 YOLO 格式 [x_center, y_center, w, h] (归一化)
    size: (width, height)
    box: [x1, y1, x2, y2]
    """
    dw = 1. / size[0]
    dh = 1. / size[1]

    # 计算中心点和宽高
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    # 归一化
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return (x, y, w, h)


def run(args):
    # 定义输出目录结构
    out_images_dir = os.path.join(args.output_dir, 'images')
    out_labels_dir = os.path.join(args.output_dir, 'labels')

    make_dirs(out_images_dir, reset=False)
    make_dirs(out_labels_dir, reset=False)

    # 获取所有标注文件
    # WiderPerson 的标注文件通常在 Annotations 文件夹下，命名如 000001.jpg.txt
    label_files = glob.glob(os.path.join(args.labels_dir_in, '*.txt'))

    logging.info(f"Found {len(label_files)} annotation files.")

    for label_file in tqdm(label_files, desc="Converting"):
        # 1. 解析文件名
        # label_file: .../Annotations/000001.jpg.txt
        filename_with_ext = os.path.basename(label_file)  # 000001.jpg.txt

        # 对应的图片文件名 (去掉 .txt)
        img_filename = filename_with_ext.replace('.txt', '')  # 000001.jpg

        # 对应的 YOLO 标签文件名 (去掉 .jpg.txt 或 .png.txt，加上 .txt)
        # 最终需要的是 000001.txt
        yolo_label_filename = os.path.splitext(img_filename)[0] + '.txt'

        img_path_in = os.path.join(args.imgs_dir_in, img_filename)
        img_path_out = os.path.join(out_images_dir, img_filename)
        label_path_out = os.path.join(out_labels_dir, yolo_label_filename)

        # 检查图片是否存在
        if not os.path.exists(img_path_in):
            logging.warning(f"Image {img_path_in} not found, skipping.")
            continue

        # 2. 读取图片获取宽高 (为了归一化)
        # 注意：如果数据集非常大，cv2.imread 可能会比较慢。
        # 如果所有图片尺寸一致，可以硬编码。但 WiderPerson 图片尺寸不一，必须读取。
        img = cv2.imread(img_path_in)
        if img is None:
            logging.warning(f"Could not read image {img_path_in}, skipping.")
            continue

        height, width, _ = img.shape

        # 3. 读取并转换标注
        yolo_lines = []
        with open(label_file, 'r') as f:
            lines = f.readlines()

            # WiderPerson 格式第一行是标注数量，跳过
            # 后续行: class_label x1 y1 x2 y2
            for line in lines[1:]:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                x1, y1, x2, y2 = parts[1], parts[2], parts[3], parts[4]

                # 坐标越界修正 (clip)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)

                # 宽度或高度无效则跳过
                if x2 <= x1 or y2 <= y1:
                    continue

                # --- 类别处理逻辑 ---
                # WiderPerson 原始定义:
                # 1: pedestrians, 2: riders, 3: partially-visible, 4: ignore, 5: crowd

                final_cls_id = -1

                if args.merge_person:
                    # 常用策略：将 1, 2, 3 合并为 0 (Person)，忽略 4 和 5
                    if cls_id in [1, 2, 3]:
                        final_cls_id = 0
                    else:
                        continue  # 忽略 ignore regions 和 crowd
                else:
                    # 默认策略：保留所有类别，ID 减 1 (转为 0-based)
                    # 0: pedestrians, 1: riders, ...
                    final_cls_id = cls_id - 1

                # 转换坐标
                bbox = (x1, y1, x2, y2)
                yolo_bbox = convert_box((width, height), bbox)

                # 格式: class x_center y_center w h
                line_content = f"{final_cls_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                yolo_lines.append(line_content)

        # 4. 写入 YOLO 格式标签文件
        if len(yolo_lines) > 0:
            with open(label_path_out, 'w') as f_out:
                f_out.write('\n'.join(yolo_lines))

            # 5. 复制图片到输出目录
            # 使用 copy2 保留文件元数据，或者 copyfile 仅复制内容
            shutil.copy(img_path_in, img_path_out)


def main():
    set_logging()
    args = parse_args()
    logging.info(f"Arguments: {args}")

    if not os.path.exists(args.imgs_dir_in) or not os.path.exists(args.labels_dir_in):
        logging.error("Input directories do not exist. Please check paths.")
        return

    logging.info("Starting conversion...")
    run(args)
    logging.info(f"Conversion completed. Data saved to {args.output_dir}")


if __name__ == '__main__':
    main()
