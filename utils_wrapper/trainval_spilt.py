import argparse
import glob
import logging
import os
import random
import shutil

from tqdm import tqdm

from utils_wrapper.utils import make_dirs


def set_logging():
    """配置简单的日志打印"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def make_div_dirs(output_root):
    """创建YOLO格式的数据集目录结构"""
    sub_dirs = [
        os.path.join(output_root, 'images', 'train'),
        os.path.join(output_root, 'images', 'val'),
        os.path.join(output_root, 'labels', 'train'),
        os.path.join(output_root, 'labels', 'val')
    ]
    for d in sub_dirs:
        os.makedirs(d, exist_ok=True)
    return sub_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset into train and val')

    # 输入路径设置
    parser.add_argument('--imgs_dir_in', type=str, default='/home/manu/tmp/person/images/train')
    parser.add_argument('--labels_dir_in', type=str, default='/home/manu/tmp/person/labels/train')

    # 输出路径设置
    parser.add_argument('--output_dir', type=str, default='/home/manu/mnt/8gpu_3090/person_sorted')

    # 参数设置
    parser.add_argument('--val_size', type=int, default=2048, help='验证集的图片数量')
    parser.add_argument('--img_ext', type=str, default='jpg', help='图片后缀名，如 jpg, png')

    return parser.parse_args()


def copy_files(file_list, args, phase):
    """
    Args:
        file_list: 图片路径列表
        args: 参数对象
        phase: 'train' 或 'val'
    """
    # 定义目标目录
    dst_img_dir = os.path.join(args.output_dir, 'images', phase)
    dst_label_dir = os.path.join(args.output_dir, 'labels', phase)

    for img_path in tqdm(file_list, desc=f"Processing {phase} set"):
        # 解析文件名
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]

        # 构造对应的 txt 标签路径
        label_name = base_name + '.txt'
        label_path = os.path.join(args.labels_dir_in, label_name)

        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            logging.warning(f"Label file not found for {file_name}, skipping...")
            continue

        # 复制图片
        shutil.copy(img_path, os.path.join(dst_img_dir, file_name))

        # 复制标签
        shutil.copy(label_path, os.path.join(dst_label_dir, label_name))


def run(args):
    make_dirs(args.output_dir, reset=True)

    # 1. 准备输出目录
    if os.path.exists(args.output_dir):
        logging.warning(f"Output directory {args.output_dir} already exists.")
    make_div_dirs(args.output_dir)

    # 2. 读取所有图片
    search_pattern = os.path.join(args.imgs_dir_in, f'*.{args.img_ext}')
    img_files = glob.glob(search_pattern)
    total_imgs = len(img_files)

    if total_imgs == 0:
        logging.error(f"No images found in {args.imgs_dir_in} with extension .{args.img_ext}")
        return

    logging.info(f"Found {total_imgs} images total.")

    # 3. 随机打乱
    random.shuffle(img_files)

    # 4. 划分数据集
    # 确保 val_size 不超过总数
    val_count = min(args.val_size, total_imgs)

    val_files = img_files[:val_count]
    train_files = img_files[val_count:]

    logging.info(f"Split result -> Train: {len(train_files)}, Val: {len(val_files)}")

    # 5. 执行复制操作
    copy_files(val_files, args, phase='val')
    copy_files(train_files, args, phase='train')


def main():
    set_logging()
    args = parse_args()

    logging.info("Arguments config:")
    print(args)

    run(args)
    logging.info("Dataset split and copy completed successfully.")


if __name__ == '__main__':
    main()
