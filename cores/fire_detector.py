import inspect
import logging
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Target:
    bbox: list = field(default_factory=list)
    last_valid_bbox: list = field(default_factory=list)  # Add this to store the last valid bbox
    bbox_list: list = field(default_factory=list)
    id: int = 0
    lost_frames: int = 0
    cls: float = 0.0
    age: int = 0
    conf_list: list = field(default_factory=list)
    diff_list: list = field(default_factory=list)
    center_list: list = field(default_factory=list)
    area_list: list = field(default_factory=list)
    area_diff_list: list = field(default_factory=list)
    mask_avg_list: list = field(default_factory=list)


class FireDetectorNight:
    def __init__(self):
        self.targets = []
        self.next_id = 0
        self.iou_threshold = 0.1
        self.max_lost_frames = 5

    def find_filtered_contours(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 4096 * 4]
        return filtered_contours

    def find_max_contour(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return max(contours, key=cv2.contourArea)
        else:
            return None

    def update(self, frame):
        contours = self.find_filtered_contours(frame)
        matched = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, x + w, y + h]
            center = [(x + x + w) / 2, (y + y + h) / 2]
            area = w * h
            best_iou = 0
            best_target = None

            for target in self.targets:
                iou = self._calculate_iou(bbox, target.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target

            if best_iou > self.iou_threshold:
                best_target.bbox = bbox
                best_target.bbox_list.append(bbox)  # Append bbox to bbox_list
                best_target.lost_frames = 0
                best_target.age += 1
                best_target.conf_list.append(1.0)  # Confidence is always 1.0 for detected contour
                best_target.center_list.append(center)
                best_target.area_list.append(area)
                if len(best_target.area_list) > 1:
                    area_diff = abs(area - best_target.area_list[-2])
                    best_target.area_diff_list.append(area_diff)
                else:
                    best_target.area_diff_list.append(0)

                if len(best_target.center_list) > 1:
                    diff = np.array(center) - np.array(best_target.center_list[-2])
                    best_target.diff_list.append(diff.tolist())  # Convert array to list
                else:
                    best_target.diff_list.append([0, 0])
                matched.append(best_target)
            else:
                new_target = Target(
                    bbox=bbox,
                    bbox_list=[bbox],  # Initialize bbox_list with the current bbox
                    id=self.next_id,
                    lost_frames=0,
                    cls=0,
                    age=1,
                    conf_list=[1.0],
                    diff_list=[[0, 0]],
                    center_list=[center],
                    area_list=[area],
                    area_diff_list=[0]
                )
                self.targets.append(new_target)
                self.next_id += 1

        for target in self.targets:
            if target not in matched:
                target.lost_frames += 1
                target.conf_list.append(0.0)
                target.diff_list.append([0, 0])
                target.area_diff_list.append(0)
                target.mask_avg_list.append(0)
                target.bbox_list.append(target.bbox)  # Ensure bbox_list is updated

        self.targets = [t for t in self.targets if t.lost_frames <= self.max_lost_frames]

    @staticmethod
    def _calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


class FireDetector:
    def __init__(self, args):
        self.opt = self._build_yolo(args)
        self.targets = []
        self.next_id = 0
        self.iou_threshold = 0.1
        self.max_lost_frames = 5
        self.previous_frame = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def infer_yolo(self, img):
        original_cwd = os.getcwd()
        res = None
        try:
            from detect_dual import run
            opt = self.opt
            opt.source_npy = img
            run_params = inspect.signature(run).parameters
            run_args = {k: v for k, v in vars(opt).items() if k in run_params}
            res = run(**run_args)
            logging.info(res)
        except Exception as e:
            logging.error(f"Error during inference: {e}")
        finally:
            os.chdir(original_cwd)
        return res

    def update(self, detections, frame):
        """
        Update targets based on new detections.

        Args:
            detections: List of detected bounding boxes in the format [x1, y1, x2, y2].
            frame: The current frame for calculating patch differences.
        """
        # Apply background subtraction
        fgmask = self.fgbg.apply(frame)

        matched = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for det in detections:
            if len(det.split()) == 6:
                cls, center_x, center_y, box_w, box_h, conf = map(float, det.split())
            else:
                cls, center_x, center_y, box_w, box_h = map(float, det.split())
                conf = 0.0

            bbox = [
                center_x - box_w / 2,
                center_y - box_w / 2,
                center_x + box_w / 2,
                center_y + box_h / 2
            ]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Calculate area of the bounding box

            best_iou = 0
            best_target = None
            for target in self.targets:
                iou = self._calculate_iou(bbox, target.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target

            if best_iou > self.iou_threshold:
                best_target.bbox = bbox
                best_target.bbox_list.append(bbox)  # Append bbox to bbox_list
                best_target.lost_frames = 0
                best_target.age += 1
                best_target.conf_list.append(conf)

                diff = self._calculate_frame_difference(
                    self.previous_frame, frame_gray, best_target.bbox, best_target.last_valid_bbox, frame.shape
                )
                best_target.diff_list.append(diff)

                # Update last valid bbox to the current bbox
                best_target.last_valid_bbox = bbox

                # Calculate mask average for the current target
                mask_avg = self._calculate_mask_avg(fgmask, best_target.bbox, frame.shape)
                best_target.mask_avg_list.append(mask_avg)

                # Update area_list and area_diff_list
                best_target.area_list.append(area)
                if len(best_target.area_list) > 1:
                    area_diff = abs(area - best_target.area_list[-2])
                    best_target.area_diff_list.append(area_diff)
                else:
                    best_target.area_diff_list.append(0.0)

                matched.append(best_target)
            else:
                new_target = Target(
                    bbox=bbox,
                    last_valid_bbox=bbox,  # Initialize last_valid_bbox with the current bbox
                    bbox_list=[bbox],  # Initialize bbox_list with the current bbox
                    id=self.next_id,
                    lost_frames=0,
                    cls=cls,
                    age=1,
                    conf_list=[conf],
                    diff_list=[0.0],
                    center_list=[(center_x, center_y)],
                    area_list=[area],
                    area_diff_list=[0.0],
                    mask_avg_list=[self._calculate_mask_avg(fgmask, bbox, frame.shape)]
                )
                self.targets.append(new_target)
                self.next_id += 1

        for target in self.targets:
            if target not in matched:
                target.lost_frames += 1
                target.conf_list.append(-1.0)
                target.diff_list.append(0.0)

                # Calculate mask average for the current target
                mask_avg = self._calculate_mask_avg(fgmask, target.bbox, frame.shape)
                target.mask_avg_list.append(mask_avg)
                target.bbox_list.append(target.bbox)  # Ensure bbox_list is updated

        self.targets = [t for t in self.targets if t.lost_frames <= self.max_lost_frames]
        self.previous_frame = frame_gray

    def _calculate_frame_difference(self, prev_frame, curr_frame, bbox, last_valid_bbox, frame_shape):
        if prev_frame is None:
            return 0.0
        curr_patch = self._extract_patch(curr_frame, bbox, frame_shape)
        if curr_patch.size == 0:
            return 0.0
        curr_center_x = (bbox[0] + bbox[2]) / 2 * frame_shape[1]
        curr_center_y = (bbox[1] + bbox[3]) / 2 * frame_shape[0]
        patch_width = curr_patch.shape[1]
        patch_height = curr_patch.shape[0]
        prev_patch = self._extract_centered_patch_with_padding(prev_frame, curr_center_x, curr_center_y, patch_width,
                                                               patch_height)
        if prev_patch.size == 0:
            return 0.0
        if prev_patch.shape != curr_patch.shape:  # TODO
            return 0.0
        return np.mean(np.abs(curr_patch - prev_patch))

    def _extract_patch(self, frame, bbox, frame_shape):
        top_left_x = int(bbox[0] * frame_shape[1])
        top_left_y = int(bbox[1] * frame_shape[0])
        bottom_right_x = int(bbox[2] * frame_shape[1])
        bottom_right_y = int(bbox[3] * frame_shape[0])

        return frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    def _extract_centered_patch_with_padding(self, frame, center_x, center_y, width, height):
        """
        从frame中提取以(center_x, center_y)为中心的patch，宽度和高度为width和height。
        如果patch超出图像边界，则进行镜像填充。
        """
        top_left_x = int(center_x - width // 2)
        top_left_y = int(center_y - height // 2)
        bottom_right_x = int(center_x + width // 2)
        bottom_right_y = int(center_y + height // 2)

        pad_left = max(0, -top_left_x)
        pad_top = max(0, -top_left_y)
        pad_right = max(0, bottom_right_x - frame.shape[1])
        pad_bottom = max(0, bottom_right_y - frame.shape[0])

        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)
        bottom_right_x = min(frame.shape[1], bottom_right_x)
        bottom_right_y = min(frame.shape[0], bottom_right_y)

        patch = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

        return patch

    @staticmethod
    def _calculate_mask_avg(mask, bbox, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return 0
        patch = mask[y1:y2, x1:x2]
        if patch.size == 0:
            return 0
        mask_sum = np.sum(patch) / 255  # Divide by 255 to count white pixels
        area = (x2 - x1) * (y2 - y1)  # Calculate the area of the patch
        return mask_sum / area  # Calculate the average mask value

    @staticmethod
    def _extract_patch(frame, bbox, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return np.array([])
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _build_yolo(self, args):
        original_cwd = os.getcwd()
        opt = None
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
            opt.model_global = DetectMultiBackend(
                args.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half
            )
        except Exception as e:
            logging.error(f"Error during YOLO model setup: {e}")
        finally:
            os.chdir(original_cwd)

        return opt
