import argparse
import logging
import os
import re

import cv2
import numpy as np

from utils_wrapper.utils import set_logging


def extract_timestamp(filename):
    # 使用正则表达式从文件名中提取时间戳
    match = re.search(r'visi_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})\.jpg', filename)
    if match:
        return match.group(1)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Process images with temporal median filter.")
    parser.add_argument('--directory', type=str, default='/home/manu/tmp/rgb2jpg/',
                        help='Directory containing image files.')
    parser.add_argument('--frame-window', type=int, default=5, help='Number of frames for median filtering.')
    parser.add_argument('--output-video', type=str, default='/home/manu/tmp/processed_output.mp4',
                        help='Output video for processed frames.')
    parser.add_argument('--output-original', type=str, default='/home/manu/tmp/original_output.mp4',
                        help='Output video for original frames.')
    return parser.parse_args()


def read_and_sort_files(directory):
    # 获取目录下的所有文件并提取时间戳进行排序
    files = os.listdir(directory)
    sorted_files = sorted(files, key=lambda x: extract_timestamp(x))
    return sorted_files


def read_image(directory, filename):
    image_path = os.path.join(directory, filename)
    image = cv2.imread(image_path)
    if image is not None:
        return image
    else:
        logging.warning(f"Failed to read image: {filename}")
        return None


def process_images_with_temporal_median(directory, sorted_files, num_frames=3, output_video='output.mp4',
                                        output_original='original_output.mp4', fps=4):
    if len(sorted_files) < num_frames:
        logging.error("Not enough images to process.")
        return

    frame_height, frame_width = cv2.imread(os.path.join(directory, sorted_files[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # VideoWriter for processed frames
    processed_out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    # VideoWriter for original frames
    original_out = cv2.VideoWriter(output_original, fourcc, fps, (frame_width, frame_height))

    for i in range(num_frames - 1, len(sorted_files)):
        frames = []
        for j in range(i - num_frames + 1, i + 1):
            image = read_image(directory, sorted_files[j])
            if image is not None:
                frames.append(image)

        # Write the current original frame to the original video
        if frames[-1] is not None:  # Ensure the last frame in the list is valid
            original_out.write(frames[-1])

        if len(frames) == num_frames:
            median_frame = np.median(frames, axis=0).astype(np.uint8)
            frame_text = f'Frame: {i + 1}'
            cv2.putText(median_frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # Write processed frame to video
            processed_out.write(median_frame)

            # Optionally display the processed frame
            cv2.imshow('Cleaned Image', median_frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

    # Release VideoWriter resources
    processed_out.release()
    original_out.release()
    cv2.destroyAllWindows()


def main():
    set_logging()
    args = parse_args()
    logging.info(f"Arguments received: {args}")

    sorted_files = read_and_sort_files(args.directory)
    if len(sorted_files) >= args.frame_window:
        process_images_with_temporal_median(args.directory, sorted_files, args.frame_window, args.output_video,
                                            args.output_original)
    else:
        logging.error("Not enough images to process.")


if __name__ == '__main__':
    main()
