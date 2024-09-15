import logging
from queue import Empty

import cv2
import numpy as np


def get_color_for_class(cls):
    # Define a mapping from class ID to color
    color_map = {
        0: (0, 255, 255),  # Green for class 0
        1: (255, 255, 0),  # Blue for class 1
        2: (255, 0, 255),  # Red for class 2
    }
    # Default to white if the class is not in the map
    return color_map.get(int(cls), (255, 255, 255))


def draw_boxes(frame, detections):
    height, width, _ = frame.shape
    for label in detections:
        # Parse the label
        if len(label.split()) == 6:
            cls, center_x, center_y, box_w, box_h, conf = map(float, label.split())
        else:
            cls, center_x, center_y, box_w, box_h = map(float, label.split())
            conf = None

        # Convert from relative coordinates to absolute coordinates
        center_x *= width
        center_y *= height
        box_w *= width
        box_h *= height

        # Calculate the top-left and bottom-right coordinates
        top_left_x = int(center_x - box_w / 2)
        top_left_y = int(center_y - box_h / 2)
        bottom_right_x = int(center_x + box_w / 2)
        bottom_right_y = int(center_y + box_h / 2)

        # Get color for the class
        color = get_color_for_class(cls)

        # Draw the rectangle and the label
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
        label_text = f"{int(cls)}"
        if conf is not None:
            label_text += f" {conf:.2f}"
        cv2.putText(frame, label_text, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def process_displayer_night(queue, queue_res, event):
    name_window = 'frame'
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)

    idx_frame_res, targets = -1, []
    while True:
        tsp_frame, idx_frame, frame, fc = queue.get()

        if idx_frame > fc - 8:
            break

        logging.info(f'displayer idx_frame --> {idx_frame}')

        while idx_frame_res < idx_frame:
            try:
                idx_frame_res, targets = queue_res.get_nowait()
                logging.info(f'displayer idx_frame_res --> {idx_frame_res}')
            except Empty:
                continue

        if idx_frame_res == idx_frame and targets is not None:
            for target in targets:
                th_age = 12
                bbox = target.bbox
                age = target.age  # Accessing age from Target dataclass
                avg_area_diff = np.mean(target.area_diff_list[-th_age:]) / target.area_list[-1]

                if age > th_age and avg_area_diff > 0.3:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame,
                            f"ID: {target.id} AGE: {target.age} LOST: {target.lost_frames}",
                            (bbox[0], bbox[1] + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                area_diff_text = f"Avg. Area Diff: {avg_area_diff:.2f}"
                cv2.putText(frame, area_diff_text, (bbox[0], bbox[1] + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            event.set()
            break

    cv2.destroyAllWindows()


def process_displayer(queue, queue_res, event):
    name_window = 'frame'
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)

    idx_frame_res, det_res, targets = -1, None, []
    while True:
        tsp_frame, idx_frame, frame, fc = queue.get()

        logging.info(f'displayer idx_frame --> {idx_frame}')

        while idx_frame_res < idx_frame:  # TODO
            try:
                idx_frame_res, det_res, targets = queue_res.get_nowait()
                logging.info(f'displayer idx_frame_res --> {idx_frame_res}')
            except Empty:
                continue

        if idx_frame_res == idx_frame and det_res is not None:
            # Extract detections from the result dictionary
            det_res = det_res.get('runs/detect/exp/labels/pseudo', [])
            # frame = draw_boxes(frame, det_res)

            th_age = 12
            # Draw tracked targets
            for target in targets:
                bbox = target.bbox  # Accessing bbox from Target dataclass
                cls = target.cls  # Accessing cls from Target dataclass
                age = target.age  # Accessing age from Target dataclass
                avg_conf = sum(target.conf_list[-th_age:]) / th_age
                avg_diff = sum(target.diff_list[-th_age:]) / th_age

                if age > th_age and avg_conf > 0.5:
                    color = (0, 0, 255)
                else:
                    color = get_color_for_class(cls)

                # Convert from relative coordinates to absolute coordinates
                top_left_x = int(bbox[0] * frame.shape[1])
                top_left_y = int(bbox[1] * frame.shape[0])
                bottom_right_x = int(bbox[2] * frame.shape[1])
                bottom_right_y = int(bbox[3] * frame.shape[0])

                cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
                cv2.putText(frame,
                            f"ID: {target.id} CLS: {int(cls)} AGE: {age} CONF: {avg_conf:.2f} DIFF: {avg_diff:.2f}",
                            (top_left_x, top_left_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            event.set()
            break

    cv2.destroyAllWindows()
