import argparse
import logging
import time
from multiprocessing import Process, Queue, Event

from processes.decoder import process_decoder
from processes.detector import process_detector
from processes.displayer import process_displayer
from utils_wrapper.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path_video',
    #                     default='/media/manu/ST8000DM004-2U91/smoke/data/ULtests/木材火/jiujing&jiachun___172.20.20.104_visi_stream-clip.mp4')
    # parser.add_argument('--path_video',
    #                     default='/media/manu/ST2000DM005-2U91/fire/data/test/V3/positive/fire (69).mp4')
    # parser.add_argument('--path_video',
    #                     default='/media/manu/ST2000DM005-2U91/fire/data/test/V3/negative/nofire (33).mp4')
    parser.add_argument('--path_video',
                        default='/media/manu/ST8000DM004-2U91/smoke/data/test/烟雾/正例（200）/smog (76).mp4')
    # parser.add_argument('--path_video',
    #                     default='/media/manu/ST2000DM005-2U91/fire/data/test/V3/negative/nofire (33).mp4')
    parser.add_argument('--source',
                        default='/media/manu/ST2000DM005-2U91/workspace/yolov9/figure/horses_prediction.jpg')
    parser.add_argument('--yolo_root', default='/media/manu/ST2000DM005-2U91/workspace/yolov9/')
    parser.add_argument('--view-img', default=False, help='show results')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--imgsz', type=int, default=1280, help='inference size h,w')
    # parser.add_argument('--weights', type=str,
    #                     default='/home/manu/mnt/8gpu_3090/test/runs/train/yolov9-s-fire-s1280_27/weights/last.pt')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size h,w')
    parser.add_argument('--weights', type=str,
                        default='/home/manu/mnt/8gpu_3090/test/runs/train_smoke/yolov9-s-smoke-s640_/weights/best.pt')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--save-txt', default=False, help='save results to *.txt')
    parser.add_argument('--nosave', default=True, help='do not save images/videos')
    parser.add_argument('--ret-res', default=True)
    parser.add_argument('--save-conf', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--alg_night', default=False)
    parser.add_argument('--save_root', type=str, default='/home/manu/tmp/fire_test_results')
    parser.add_argument('--show', default=False)
    return parser.parse_args()


def run(args):
    logging.info(args)

    stop_event = Event()

    q_detector = Queue()
    q_detector_res = Queue()
    p_detector = Process(target=process_detector, args=(args, q_detector, q_detector_res, stop_event), daemon=True)
    p_detector.start()
    time.sleep(4)  # wait for model init

    q_displayer = Queue()
    p_displayer = Process(
        target=process_displayer,
        args=(q_displayer, q_detector_res, stop_event, args.path_video, args.show, args.save_root),
        daemon=True)
    p_displayer.start()

    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(args.path_video, q_decoder, stop_event), daemon=True)
    p_decoder.start()

    while True:
        item_frame = q_decoder.get()
        tsp_frame, idx_frame, frame, fc = item_frame
        if frame is None or stop_event.is_set():
            break
        q_detector.put(item_frame)
        q_displayer.put(item_frame)

    logging.info('main processing loop exited gracefully.')


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
