import argparse
import glob
import os
import shutil

from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_dir', default='/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/vis-10-09/正例')
    parser.add_argument('--neg_dir', default='/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/vis-10-09/反例')
    parser.add_argument('--vis_dir',
                        default='/media/manu/ST2000DM005-2U91/fire/data/pseudo/fire_unlabeled/VOCdevkit/VOC2007/JPEGImages')
    parser.add_argument('--preds_dir',
                        default='/media/manu/ST2000DM005-2U91/fire/mixpl/results/fire_unlabeled_res_all/preds')
    parser.add_argument('--output_imgs_dir', default='/home/manu/tmp/mm_filtered_res/imgs')
    parser.add_argument('--output_labels_dir', default='/home/manu/tmp/mm_filtered_res/labels')
    return parser.parse_args()


def copy_positive_examples(pos_dir, vis_dir, preds_dir, output_imgs_dir, output_labels_dir):
    pos_files = glob.glob(os.path.join(pos_dir, '*.jpg'))

    for file_path in tqdm(pos_files, desc="Processing positive examples"):
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]

        # Copy image to imgs directory
        original_img_path = os.path.join(vis_dir, file_name)
        if os.path.exists(original_img_path):
            shutil.copy2(original_img_path, os.path.join(output_imgs_dir, file_name))

        # Copy corresponding txt file to labels directory
        pred_txt_path = os.path.join(preds_dir, base_name + '.json')
        if os.path.exists(pred_txt_path):
            shutil.copy2(pred_txt_path, os.path.join(output_labels_dir, base_name + '.json'))


def copy_negative_examples(neg_dir, vis_dir, output_imgs_dir):
    neg_files = glob.glob(os.path.join(neg_dir, '*.jpg'))

    for file_path in tqdm(neg_files, desc="Processing negative examples"):
        file_name = os.path.basename(file_path)

        # Copy image to imgs directory
        original_img_path = os.path.join(vis_dir, file_name)
        if os.path.exists(original_img_path):
            shutil.copy2(original_img_path, os.path.join(output_imgs_dir, file_name))


def main():
    set_logging()
    args = parse_args()

    make_dirs(args.output_imgs_dir, reset=True)
    make_dirs(args.output_labels_dir, reset=True)

    # Copy files based on the criteria
    copy_positive_examples(args.pos_dir, args.vis_dir, args.preds_dir, args.output_imgs_dir, args.output_labels_dir)
    copy_negative_examples(args.neg_dir, args.vis_dir, args.output_imgs_dir)


if __name__ == '__main__':
    main()
