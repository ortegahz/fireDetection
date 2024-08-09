import argparse
import logging
import os
import sys

import inspect

sys.path.append(os.getcwd())
from utils_wrapper.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_root', default='/media/manu/data/workspace/yolov9/')
    parser.add_argument('--source', type=str,
                        default='/home/manu/tmp/BOSH-FM数据采集/zheng-shiwai/Z-D-80m-002.mp4')
    parser.add_argument('--imgsz', type=int, default=1280, help='inference size h,w')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str,
                        default='/run/user/1000/gvfs/smb-share:server=172.20.254.132,share=sharedfolder/test/runs/train/yolov9-s-fire-s1280_3/weights/best.pt')
    parser.add_argument('--name', type=str, default='manu_detect')
    parser.add_argument('--view-img', default=True, help='show results')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--save-txt', default=True, help='save results to *.txt')
    parser.add_argument('--project', default='/home/manu/tmp/runs', help='save results to project/name')
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
        run_params = inspect.signature(run).parameters
        run_args = {k: v for k, v in vars(opt).items() if k in run_params}
        run(**run_args)
    finally:
        os.chdir(original_cwd)


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run_wrapper(args)


if __name__ == '__main__':
    main()
