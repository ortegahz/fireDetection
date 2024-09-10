import inspect
import logging
import os
import sys

from queue import Empty


def process_detector(args, queue, queue_res, event):
    original_cwd = os.getcwd()
    try:
        os.chdir(args.yolo_root)
        sys.path.append(args.yolo_root)
        from detect_dual import parse_opt, run
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device
        opt = parse_opt()
        opt.__dict__.update(vars(args))
        opt.imgsz = [args.imgsz, args.imgsz]
        device = select_device(opt.device)
        opt.model_global = DetectMultiBackend(args.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
    finally:
        os.chdir(original_cwd)

    latest_item = None
    while True:
        # Get the latest frame from the queue
        while True:
            try:
                latest_item = queue.get_nowait()
            except Empty:
                break

        if not latest_item:
            continue

        tsp_frame, idx_frame, frame, fc = latest_item

        # logging.info(f'detector idx_frame --> {idx_frame}')

        # TODO: process
        original_cwd = os.getcwd()
        try:
            # opt.source_npy = cv2.imread(opt.source)
            opt.source_npy = frame
            run_params = inspect.signature(run).parameters
            run_args = {k: v for k, v in vars(opt).items() if k in run_params}
            res = run(**run_args)
            logging.info(res)
            queue_res.put((idx_frame, res))
        finally:
            os.chdir(original_cwd)

        if event.is_set():
            break
