import logging
import os
from queue import Empty

import cv2
import numpy as np


def get_color_for_class(cls):
    color_map = {
        0: (128, 255, 128),  # Green for class 0
        1: (255, 255, 0),  # Blue for class 1
        2: (255, 0, 255),  # Red for class 2
    }
    return color_map.get(int(cls), (255, 255, 255))


def draw_boxes(frame, detections):
    height, width, _ = frame.shape
    for label in detections:
        if len(label.split()) == 6:
            cls, center_x, center_y, box_w, box_h, conf = map(float, label.split())
        else:
            cls, center_x, center_y, box_w, box_h = map(float, label.split())
            conf = None

        center_x *= width
        center_y *= height
        box_w *= width
        box_h *= height

        top_left_x = int(center_x - box_w / 2)
        top_left_y = int(center_y - box_w / 2)
        bottom_right_x = int(center_x + box_w / 2)
        bottom_right_y = int(center_y + box_w / 2)

        color = get_color_for_class(cls)

        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
        label_text = f"{int(cls)}"
        if conf is not None:
            label_text += f" {conf:.2f}"
        cv2.putText(frame, label_text, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def calculate_normalized_distance(initial_bbox, current_bbox, frame_shape):
    initial_center = [(initial_bbox[0] + initial_bbox[2]) / 2 * frame_shape[1],
                      (initial_bbox[1] + initial_bbox[3]) / 2 * frame_shape[0]]
    current_center = [(current_bbox[0] + current_bbox[2]) / 2 * frame_shape[1],
                      (current_bbox[1] + current_bbox[3]) / 2 * frame_shape[0]]
    euclidean_distance = np.sqrt((initial_center[0] - current_center[0]) ** 2 +
                                 (initial_center[1] - current_center[1]) ** 2)

    top_left_x = int(current_bbox[0] * frame_shape[1])
    top_left_y = int(current_bbox[1] * frame_shape[0])
    bottom_right_x = int(current_bbox[2] * frame_shape[1])
    bottom_right_y = int(current_bbox[3] * frame_shape[0])

    current_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    normalized_distance = euclidean_distance / current_area
    return normalized_distance


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
                th_age = 8
                bbox = target.bbox
                age = target.age
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

                # Draw the trajectory using the stored positions in bbox_list
                for i in range(1, len(target.bbox_list)):
                    prev_bbox = target.bbox_list[i - 1]
                    curr_bbox = target.bbox_list[i]
                    prev_center = (int((prev_bbox[0] + prev_bbox[2]) / 2), int((prev_bbox[1] + prev_bbox[3]) / 2))
                    curr_center = (int((curr_bbox[0] + curr_bbox[2]) / 2), int((curr_bbox[1] + curr_bbox[3]) / 2))
                    cv2.line(frame, prev_center, curr_center, color, 2)

                # Calculate the normalized distance
                normalized_distance = calculate_normalized_distance(target.bbox_list[0], bbox, frame.shape)
                distance_text = f"Norm Dist: {normalized_distance:.2f}"
                cv2.putText(frame, distance_text, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            event.set()
            break

    cv2.destroyAllWindows()


def process_displayer(queue, queue_res, event,
                      video_path='/media/manu/ST2000DM005-2U91/fire/test/V3/negative/nofire (4096).mp4', show=True,
                      save_root='/home/manu/tmp/fire_test_results'):
    video_name = os.path.basename(video_path)
    if show:
        name_window = 'frame'
        cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    idx_frame_res, det_res, targets, is_alarm = -1, None, [], False
    while True:
        tsp_frame, idx_frame, frame, fc = queue.get()

        logging.info(f'displayer idx_frame --> {idx_frame}')

        while idx_frame_res < idx_frame:
            try:
                idx_frame_res, det_res, targets = queue_res.get_nowait()
                logging.info(f'displayer idx_frame_res --> {idx_frame_res}')
            except Empty:
                continue

        if idx_frame_res == idx_frame and det_res is not None:
            det_res = det_res.get('runs/detect/exp/labels/pseudo', [])
            # frame = draw_boxes(frame, det_res)

            th_age = 8
            for target in targets:
                bbox = target.bbox
                cls = target.cls
                age = target.age
                avg_conf = sum(target.conf_list[-th_age:]) / th_age
                # avg_diff = sum(target.diff_list[-th_age:]) / th_age
                mask_avg = sum(target.mask_avg_list[-th_age:]) / th_age

                top_left_x = int(bbox[0] * frame.shape[1])
                top_left_y = int(bbox[1] * frame.shape[0])
                bottom_right_x = int(bbox[2] * frame.shape[1])
                bottom_right_y = int(bbox[3] * frame.shape[0])

                # Calculate the normalized distance
                normalized_distance = calculate_normalized_distance(target.bbox_list[0], bbox, frame.shape) * 16
                distance_text = f"Norm Dist: {normalized_distance:.2f}"

                if th_age < age < th_age * 16 and avg_conf > 0.5 and mask_avg > 0.3 and normalized_distance < 0.5:  # noqa
                    color = (0, 0, 255)
                    is_alarm = True
                    print(f'is_alarm --> {is_alarm}')
                    alarm_status = "ALARM" if is_alarm else "NO ALARM"
                    with open(os.path.join(save_root, f'{video_name}.txt'), 'w') as f:
                        f.write(f'{video_name} <{alarm_status}>\n')
                    event.set()
                else:
                    color = get_color_for_class(cls)

                cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
                cv2.putText(frame,
                            f"I{target.id} A{age} C{avg_conf:.2f} M{mask_avg:.2f}",
                            (top_left_x, top_left_y + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                cv2.putText(frame, distance_text, (top_left_x, top_left_y + 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Draw the trajectory using the stored positions in bbox_list
                for i in range(1, len(target.bbox_list)):
                    prev_bbox = target.bbox_list[i - 1]
                    curr_bbox = target.bbox_list[i]
                    prev_center = (int((prev_bbox[0] + prev_bbox[2]) / 2 * frame.shape[1]),
                                   int((prev_bbox[1] + prev_bbox[3]) / 2 * frame.shape[0]))
                    curr_center = (int((curr_bbox[0] + curr_bbox[2]) / 2 * frame.shape[1]),
                                   int((curr_bbox[1] + curr_bbox[3]) / 2 * frame.shape[0]))
                    cv2.line(frame, prev_center, curr_center, color, 2)

        alarm_status = "ALARM" if is_alarm else "NO ALARM"
        with open(os.path.join(save_root, f'{video_name}.txt'), 'w') as f:
            f.write(f'{video_name} <{alarm_status}>\n')

        if show:
            cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            event.set()
            break
        if event.is_set():
            break

    cv2.setWindowProperty(name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.destroyAllWindows()

    logging.info('displayer processing loop exited gracefully.')
