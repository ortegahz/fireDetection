from queue import Empty

from cores.fire_detector import FireDetector


def process_detector(args, queue, queue_res, event):
    fire_detector = FireDetector(args)

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
        res = fire_detector.infer_yolo(frame)
        queue_res.put((idx_frame, res))

        if event.is_set():
            break
