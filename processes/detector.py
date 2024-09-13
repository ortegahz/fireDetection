import logging
from queue import Empty

from cores.fire_detector import FireDetector, FireDetectorNight


def process_detector_night(queue, queue_res, event):
    gray_frame_detector = FireDetectorNight()

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

        max_contour = gray_frame_detector.find_max_contour(frame)

        # Add detection results to the output
        queue_res.put((idx_frame, max_contour))

    logging.info('Processing loop exited gracefully.')


def process_detector(args, queue, queue_res, event):
    fire_detector = FireDetector(args)

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
        res = fire_detector.infer_yolo(frame)

        # Update targets with detection results
        detections = res.get('runs/detect/exp/labels/pseudo', [])
        fire_detector.update(detections, frame)

        # Add target tracking results to the output
        queue_res.put((idx_frame, res, fire_detector.targets))

    logging.info('Processing loop exited gracefully.')
