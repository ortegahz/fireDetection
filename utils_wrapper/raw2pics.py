import argparse
import glob
import logging
import os

import cv2
from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_in', default='/media/manu/ST8000DM004-2U91/jb_raw/03数据标注/')
    parser.add_argument('--dir_root_out', default='/media/manu/ST8000DM004-2U91/jb_raw/03数据标注-samples/')
    parser.add_argument('--sample_interval', default=8, type=int)
    parser.add_argument('--image_sample_interval', default=1, type=int)
    return parser.parse_args()


def extract_frames(video_path, output_path, interval=4):
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    saved_frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    estimated_frames_to_save = total_frames // frame_interval

    with tqdm(total=estimated_frames_to_save, desc=os.path.basename(video_path), unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_output_path = os.path.join(output_path, f"frame_{saved_frame_count:06d}.jpg")
                cv2.imwrite(frame_output_path, frame)
                saved_frame_count += 1
                pbar.update(1)
            frame_count += 1

    cap.release()
    logging.info(f"Finished processing video: {video_path}. Total saved frames: {saved_frame_count}")


def sample_images(image_paths, output_path, interval=10):
    logging.info(f"Sampling images in directory: {output_path}")
    make_dirs(output_path, reset=False)
    for idx, image_path in enumerate(tqdm(image_paths, desc="Sampling images", unit='image')):
        if idx % interval == 0:
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Failed to read image file: {image_path}")
                continue
            image_name = os.path.basename(image_path)
            output_image_path = os.path.join(output_path, image_name)
            cv2.imwrite(output_image_path, img)
    logging.info(f"Finished sampling images. Total sampled images: {len(image_paths) // interval}")


def run(args):
    make_dirs(args.dir_root_out, reset=True)
    video_extensions = ['*.mp4', '*.avi', '*.mov']
    image_extensions = ['*.jpg', '*.jpeg', '*.png']

    video_paths = []
    image_paths = []

    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(args.dir_root_in, '**', ext), recursive=True))

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.dir_root_in, '**', ext), recursive=True))

    total_videos = len(video_paths)
    total_images = len(image_paths)

    logging.info(f"Found {total_videos} video files.")
    logging.info(f"Found {total_images} image files.")

    # Process Videos
    for idx, video_path_in in enumerate(video_paths, 1):
        logging.info(f"Processing video {idx}/{total_videos}: {video_path_in}")
        relative_path = os.path.relpath(video_path_in, args.dir_root_in)
        relative_path = os.path.dirname(relative_path)
        video_name = os.path.splitext(os.path.basename(video_path_in))[0]
        output_dir = os.path.join(args.dir_root_out, relative_path, video_name)
        if len(output_dir) > 128:
            continue
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        extract_frames(video_path_in, output_dir, interval=args.sample_interval)
        logging.info(f"Finished video {idx}/{total_videos}")

    # Process Images
    if total_images > 0:
        relative_path = os.path.relpath(args.dir_root_in, args.dir_root_in)
        output_dir = os.path.join(args.dir_root_out, relative_path, "sampled_images")
        sample_images(image_paths, output_dir, interval=args.image_sample_interval)


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
