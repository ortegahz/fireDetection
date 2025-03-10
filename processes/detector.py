import logging
from multiprocessing import Queue

from cores.x_detector import *


def process_detector_night(queue, queue_res, event):
    fire_detector_night = FireDetectorNight()

    while not event.is_set():
        latest_item = None
        # Get the latest frame from the queue
        while True:
            try:
                latest_item = queue.get_nowait()
            except Empty:
                break

        if latest_item is None:
            continue

        tsp_frame, idx_frame, frame, fc = latest_item

        logging.info(f'Detector idx_frame --> {idx_frame}')

        fire_detector_night.update(frame)

        # Add detection results to the output
        queue_res.put((idx_frame, fire_detector_night.targets))

    logging.info('Processing loop exited gracefully.')


def process_detector(args, queue, queue_res, event):
    _detector = FireDetector(args)
    # _detector = SmokeDetector(args)

    while not event.is_set():
        latest_item = None
        while queue.qsize() > 0:
            latest_item = queue.get()

        if latest_item is None:
            continue

        tsp_frame, idx_frame, frame, fc = latest_item

        logging.info(f'Detector idx_frame --> {idx_frame} / {queue.qsize()}')
        # res = _detector.infer_yolo(frame)
        # detections = res.get('runs/detect/exp/labels/pseudo', [])
        detections = _detector.infer_yolo11(frame)
        _detector.update(detections, frame, None)

        # Add target tracking results to the output
        queue_res.put((idx_frame, detections, _detector.targets, _detector.frame_buffer))

    logging.info('detector processing loop exited gracefully.')
