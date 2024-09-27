import argparse
import glob
import logging
import os
import time
from multiprocessing import Process, Queue, Event
from queue import Empty

# import your processes and utils here
from processes.decoder import process_decoder
from processes.detector import process_detector
from processes.displayer import process_displayer
from utils_wrapper.utils import make_dirs
from utils_wrapper.utils import set_logging


def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except Empty:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder',
                        default='/media/manu/ST2000DM005-2U91/fire/test/V3/positive/',
                        help='Path to the folder containing videos')
    parser.add_argument('--source',
                        default='/media/manu/ST2000DM005-2U91/workspace/yolov9/figure/horses_prediction.jpg')
    parser.add_argument('--yolo_root', default='/media/manu/ST2000DM005-2U91/workspace/yolov9/')
    parser.add_argument('--view-img', default=False, help='show results')
    parser.add_argument('--imgsz', type=int, default=1280, help='inference size h,w')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str,
                        default='/home/manu/mnt/8gpu_3090/test/runs/train/yolov9-s-fire-s1280_10/weights/last.pt')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--save-txt', default=False, help='save results to *.txt')
    parser.add_argument('--nosave', default=True, help='do not save images/videos')
    parser.add_argument('--ret-res', default=True)
    parser.add_argument('--save-conf', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--alg_night', default=False)
    parser.add_argument('--save_root', default='/home/manu/tmp/fire_test_results')
    return parser.parse_args()


def get_video_files(video_folder):
    video_extensions = ["*.mp4", "*.avi", "*.mov"]  # Add more extensions as needed
    video_files = []
    for ext in video_extensions:
        # Use glob with recursive option
        video_files.extend(glob.glob(os.path.join(video_folder, '**', ext), recursive=True))
    return video_files


def run(args, video_file):
    logging.info(args)

    stop_event = Event()

    q_detector = Queue()
    q_detector_res = Queue()
    p_detector = Process(target=process_detector, args=(args, q_detector, q_detector_res, stop_event), daemon=True)
    p_detector.start()
    time.sleep(3)  # wait for model init

    q_displayer = Queue()

    p_displayer.start()

    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(video_file, q_decoder, stop_event), daemon=True)
    p_decoder.start()

    while True:
        item_frame = q_decoder.get()
        tsp_frame, idx_frame, frame, fc = item_frame
        if frame is None or stop_event.is_set():
            break
        q_detector.put(item_frame)
        q_displayer.put(item_frame)

    p_displayer.join()
    p_decoder.join()
    p_detector.join()

    logging.info('main_multi processing loop exited gracefully.')


def main():
    set_logging()
    args = parse_args()
    video_files = get_video_files(args.video_folder)
    make_dirs(args.save_root, reset=True)

    for video_file in video_files:
        logging.info(f"Processing video: {video_file}")
        run(args, video_file)
        logging.info(f"Finished processing video: {video_file}")


if __name__ == '__main__':
    main()
