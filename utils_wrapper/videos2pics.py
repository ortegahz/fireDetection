import argparse
import glob
import logging
import os

import cv2
from tqdm import tqdm

from utils import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_in', default='/home/manu/tmp/BOSH-FM数据采集/')
    parser.add_argument('--dir_root_out', default='/home/manu/tmp/BOSH-FM数据采集-samples/')
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


def run(args):
    make_dirs(args.dir_root_out, reset=True)
    video_extensions = ['*.mp4', '*.avi', '*.mov']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(args.dir_root_in, '**', ext), recursive=True))

    total_videos = len(video_paths)
    logging.info(f"Found {total_videos} video files.")

    for idx, video_path_in in enumerate(video_paths, 1):
        logging.info(f"Processing video {idx}/{total_videos}: {video_path_in}")
        relative_path = os.path.relpath(video_path_in, args.dir_root_in)
        relative_path = os.path.dirname(relative_path)
        video_name = os.path.splitext(os.path.basename(video_path_in))[0]
        output_dir = os.path.join(args.dir_root_out, relative_path, video_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        extract_frames(video_path_in, output_dir)
        logging.info(f"Finished video {idx}/{total_videos}")


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
