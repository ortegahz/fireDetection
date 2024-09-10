import logging
import time

import cv2


def process_decoder(path_video, queue, event, buff_len=5, fps_scale=4):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    logging.info(f'fps --> {fps}')
    if not cap.isOpened():
        logging.error('failed to open video stream !')

    t_last = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        tsp_frame = time.time()
        if not ret:
            queue.put([tsp_frame, idx_frame, None, fc])
            logging.warning('decoder exiting !')
            event.set()
            break

        idx_frame += 1

        if not idx_frame % fps_scale == 0:
            continue

        if queue.qsize() > buff_len:
            queue.get()
            logging.warning('dropping frame !')
        queue.put([tsp_frame, idx_frame, frame, fc])
        # logging.info(f'decoder idx_frame --> {idx_frame}')

        while time.time() - t_last < 1. / (fps / fps_scale):
            time.sleep(0.001)
        t_last = time.time()

    cap.release()