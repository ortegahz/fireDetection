import argparse
import glob
import logging
import os
import random
import shutil

from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/home/manu/tmp/cls_fire_raw',
                        help='Path to the input directory containing pos and neg subdirectories.')
    parser.add_argument('--output_dir', default='/home/manu/tmp/cls_fire',
                        help='Path to the output directory to save train and val subdirectories.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of validation set size to the total dataset size.')
    return parser.parse_args()


def split_data(input_dir, output_dir, val_ratio):
    pos_dir = os.path.join(input_dir, 'pos')
    neg_dir = os.path.join(input_dir, 'neg')

    pos_files = glob.glob(os.path.join(pos_dir, '*'))
    neg_files = glob.glob(os.path.join(neg_dir, '*'))

    random.shuffle(pos_files)
    random.shuffle(neg_files)

    # neg_files = random.sample(neg_files, len(pos_files))  # assert len() > len(pos_files)
    # pos_files = random.sample(pos_files, len(neg_files))  # assert len(pos_files) > len(neg_files)

    val_size_pos = int(len(pos_files) * val_ratio)
    val_size_neg = int(len(neg_files) * val_ratio)

    pos_train_files = pos_files[val_size_pos:]
    pos_val_files = pos_files[:val_size_pos]

    neg_train_files = neg_files[val_size_neg:]
    neg_val_files = neg_files[:val_size_neg]

    train_pos_dir = os.path.join(output_dir, 'train', 'pos')
    train_neg_dir = os.path.join(output_dir, 'train', 'neg')
    val_pos_dir = os.path.join(output_dir, 'val', 'pos')
    val_neg_dir = os.path.join(output_dir, 'val', 'neg')

    make_dirs(output_dir, reset=True)
    make_dirs(train_pos_dir, reset=True)
    make_dirs(train_neg_dir, reset=True)
    make_dirs(val_pos_dir, reset=True)
    make_dirs(val_neg_dir, reset=True)

    for file in tqdm(pos_train_files, desc="Copying pos train files"):
        shutil.copy(file, train_pos_dir)

    for file in tqdm(pos_val_files, desc="Copying pos val files"):
        shutil.copy(file, val_pos_dir)

    for file in tqdm(neg_train_files, desc="Copying neg train files"):
        shutil.copy(file, train_neg_dir)

    for file in tqdm(neg_val_files, desc="Copying neg val files"):
        shutil.copy(file, val_neg_dir)

    logging.info(f"Copied {len(pos_train_files)} pos train files and {len(pos_val_files)} pos val files.")
    logging.info(f"Copied {len(neg_train_files)} neg train files and {len(neg_val_files)} neg val files.")


def main():
    set_logging()
    args = parse_args()
    logging.info(f"Arguments: {args}")
    split_data(args.input_dir, args.output_dir, args.val_ratio)
    logging.info("Operation completed.")


if __name__ == '__main__':
    main()
