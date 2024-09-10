import inspect
import logging
import os
import sys


class FireDetector:
    def __init__(self, args):
        self.opt = self._build_yolo(args)

    def infer_yolo(self, img):
        original_cwd = os.getcwd()
        try:
            from detect_dual import run
            opt = self.opt
            opt.source_npy = img
            run_params = inspect.signature(run).parameters
            run_args = {k: v for k, v in vars(opt).items() if k in run_params}
            res = run(**run_args)
            logging.info(res)
        finally:
            os.chdir(original_cwd)
        return res

    def update(self):
        pass

    def _build_yolo(self, args):
        original_cwd = os.getcwd()
        try:
            os.chdir(args.yolo_root)
            sys.path.append(args.yolo_root)
            from detect_dual import parse_opt
            from models.common import DetectMultiBackend
            from utils.torch_utils import select_device
            opt = parse_opt()
            opt.__dict__.update(vars(args))
            opt.imgsz = [args.imgsz, args.imgsz]
            device = select_device(opt.device)
            opt.model_global = DetectMultiBackend(args.weights, device=device, dnn=opt.dnn, data=opt.data,
                                                  fp16=opt.half)
        finally:
            os.chdir(original_cwd)

        return opt
