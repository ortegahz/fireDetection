import logging
import os

import cv2
import numpy as np


def _patch_save(target, idx_frame, video_name, save_root):
    # Save combined HSV-H patch
    if target.flow_patches and target.diff_patches and target.image_patches:
        image_patch = target.image_patches[-1]
        image_patch_file_path = os.path.join(save_root,
                                             f'{video_name}_image_patch_{target.id}_{idx_frame}.jpg')
        cv2.imwrite(image_patch_file_path, image_patch)

        flow_patch = target.flow_patches[-1]
        diff_patch = target.diff_patches[-1].astype(np.uint8)
        deltax = flow_patch[..., 0]
        deltay = flow_patch[..., 1]

        # Convert the image patch to HSV and get the hue channel
        hsv_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_patch[..., 0]

        # Compute the magnitude of the flow vectors
        flow_magnitude = cv2.magnitude(deltax, deltay).astype(np.uint8)

        # Create an HSV image by merging the hue, diff (as saturation), and flow magnitude (as value)
        combined_display = cv2.merge((hue_channel, diff_patch, flow_magnitude))

        # Save the combined image
        combined_file_path = os.path.join(save_root,
                                          f'{video_name}_combined_patch_{target.id}_{idx_frame}.jpg')
        cv2.imwrite(combined_file_path, combined_display)


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


def calculate_avg_area_diff(target, th_age):
    if len(target.area_list) < th_age + 1:
        return 0.0
    diffs = [abs(target.area_list[i] - target.area_list[i - 1]) for i in range(-th_age, 0)]
    return sum(diffs) / len(diffs)


def process_displayer(queue, queue_res, event,
                      video_path='/media/manu/ST2000DM005-2U91/fire/test/V3/negative/nofire (4096).mp4', show=True,
                      save_root='/home/manu/tmp/fire_test_results', is_sample=False):
    video_name = os.path.basename(video_path)
    if show:
        name_window = 'frame'
        cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # _config = '/home/manu/mnt/ST2000DM005-2U91/workspace/mmpretrain/configs/resnet/resnet18_8xb32_fire.py'
    # _checkpoint = '/home/manu/tmp/work_dirs/resnet18_8xb32_fire/epoch_100.pth'
    # model_cls = ImageClassificationInferencer(model=_config, pretrained=_checkpoint, device='cuda')

    idx_frame_res, det_res, targets, is_alarm = -1, None, [], False
    last_idx_frame = -1
    while not event.is_set():
        tsp_frame, idx_frame, frame, fc = queue.get()

        logging.info(f'displayer idx_frame --> {idx_frame} / {queue.qsize()}')

        while idx_frame_res < idx_frame and not event.is_set():
            idx_frame_res, det_res, targets = queue_res.get()
            logging.info(f'displayer idx_frame_res --> {idx_frame_res} / {queue_res.qsize()}')

        if last_idx_frame == idx_frame:
            event.set()
            break

        if idx_frame_res == idx_frame and det_res is not None:
            last_idx_frame = idx_frame
            det_res = det_res.get('runs/detect/exp/labels/pseudo', [])

            th_age = 2 if is_sample else 4
            for target in targets:
                bbox = target.bbox
                cls = target.cls
                age = target.age
                area = target.area_list[-1]
                avg_conf = sum(target.conf_list[-th_age:]) / th_age
                avg_conf_cls = sum(target.conf_cls_list[-th_age:]) / th_age
                avg_conf_cls_rgb = sum(target.conf_cls_rgb_list[-th_age:]) / th_age
                avg_diff = sum(target.diff_list[-th_age:]) / th_age
                mask_avg = sum(target.mask_avg_list[-th_age:]) / th_age
                depth_avg = target.depth_avg_list[-1]
                avg_area_diff = sum(target.area_diff_list[-th_age:]) / th_age / area if age > th_age else 0.0
                avg_flow_consistency = sum(target.flow_consistency_list[-th_age:]) / th_age if age > th_age else 0.0

                top_left_x = int(bbox[0] * frame.shape[1])
                top_left_y = int(bbox[1] * frame.shape[0])
                bottom_right_x = int(bbox[2] * frame.shape[1])
                bottom_right_y = int(bbox[3] * frame.shape[0])

                normalized_distance = calculate_normalized_distance(target.bbox_list[-th_age], bbox,
                                                                    frame.shape) * 16 if th_age < age else 0.0

                # conf_cls = _patch_infer(target, model_cls)

                if is_sample:
                    cdt = age > th_age and avg_conf > 0.2 and idx_frame > 25 * 4 and mask_avg > 0.4  # for pos sampling
                    # cdt = age > th_age and avg_conf > 0.2  # for neg sampling
                else:
                    cdt = age > th_age and avg_conf > 0.2 and avg_conf_cls > 0.4 and mask_avg > 0.1

                color = get_color_for_class(cls)

                # Draw the trajectory using the stored positions in bbox_list
                for i in range(1, len(target.bbox_list)):
                    prev_bbox = target.bbox_list[i - 1]
                    curr_bbox = target.bbox_list[i]
                    prev_center = (int((prev_bbox[0] + prev_bbox[2]) / 2 * frame.shape[1]),
                                   int((prev_bbox[1] + prev_bbox[3]) / 2 * frame.shape[0]))
                    curr_center = (int((curr_bbox[0] + curr_bbox[2]) / 2 * frame.shape[1]),
                                   int((curr_bbox[1] + curr_bbox[3]) / 2 * frame.shape[0]))
                    cv2.line(frame, prev_center, curr_center, color, 2)

                if cdt:
                    color = (0, 0, 255)
                    is_alarm = True
                    print(f'is_alarm --> {is_alarm}')

                    if is_sample:
                        _patch_save(target, idx_frame, video_name, save_root)
                    else:
                        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
                        frame_file_path = os.path.join(save_root, f'{video_name}.jpg')
                        cv2.imwrite(frame_file_path, frame)
                        # event.set()

                cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
                cv2.putText(frame,
                            f"I{target.id} A{age} C{avg_conf:.2f} F{avg_conf_cls:.2f} M{mask_avg:.2f}",
                            (top_left_x, top_left_y + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame,
                            f"D{depth_avg:.2f}",
                            (top_left_x, top_left_y + 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if not is_sample:
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

    if show:
        cv2.setWindowProperty(name_window, cv2.WINDOW_NORMAL)
        cv2.destroyAllWindows()

    logging.info('displayer processing loop exited gracefully.')
