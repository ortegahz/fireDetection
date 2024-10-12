import argparse
import logging
import os
import sys
import cv2

import inspect

sys.path.append(os.getcwd())
from utils_wrapper.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_root', default='/media/manu/ST2000DM005-2U91/workspace/yolov9/')
    # parser.add_argument('--source', type=str,
    #                     default='/home/manu/tmp/1/01 数据采集_bosch数据采集_博世数据_博世数据采集20240614_huo-shinei_huo-4mm-10m-0_frame_000000.jpg')
    # parser.add_argument('--view-img', default=False, help='show results')
    parser.add_argument('--source', type=str,
                        default='/home/manu/tmp/vlc-record-2024-09-18-20h01m17s-rtsp___172.20.20.102_visi_stream-.mp4')
    parser.add_argument('--view-img', default=True, help='show results')
    parser.add_argument('--imgsz', type=int, default=1280, help='inference size h,w')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str,
                        default='/home/manu/mnt/8gpu_3090/test/runs/train/yolov9-s-fire-s1280_11/weights/best.pt')
    parser.add_argument('--name', type=str, default='manu_detect')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--save-txt', default=True, help='save results to *.txt')
    parser.add_argument('--ret-res', default=True)
    parser.add_argument('--save-conf', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--project', default='/media/manu/ST8000DM004-2U91/runs', help='save results to project/name')
    return parser.parse_args()


def run_wrapper(args):
    original_cwd = os.getcwd()
    try:
        os.chdir(args.yolo_root)
        sys.path.append(args.yolo_root)
        from detect_dual import parse_opt, run
        opt = parse_opt()
        opt.__dict__.update(vars(args))
        opt.imgsz = [args.imgsz, args.imgsz]
        opt.source_npy = cv2.imread(opt.source)
        run_params = inspect.signature(run).parameters
        run_args = {k: v for k, v in vars(opt).items() if k in run_params}
        res = run(**run_args)
        logging.info(res)
    finally:
        os.chdir(original_cwd)


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run_wrapper(args)


if __name__ == '__main__':
    main()
