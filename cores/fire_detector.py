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
    id: int = 0
    lost_frames: int = 0
    cls: float = 0.0
    age: int = 0
    conf_list: list = field(default_factory=list)
    diff_list: list = field(default_factory=list)
    center_list: list = field(default_factory=list)


class FireDetectorNight:
    def __init__(self):
        self.targets = []
        self.next_id = 0
        self.iou_threshold = 0.1
        self.max_lost_frames = 5

    def find_max_contour(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return max(contours, key=cv2.contourArea)
        else:
            return None

    def update(self, frame):
        max_contour = self.find_max_contour(frame)
        matched = []

        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)
            bbox = [x, y, x + w, y + h]
            center = [(x + x + w) / 2, (y + y + h) / 2]
            best_iou = 0
            best_target = None

            for target in self.targets:
                iou = self._calculate_iou(bbox, target.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target

            if best_iou > self.iou_threshold:
                best_target.bbox = bbox
                best_target.lost_frames = 0
                best_target.age += 1
                best_target.conf_list.append(1.0)  # Confidence is always 1.0 for detected contour
                best_target.center_list.append(center)
                if len(best_target.center_list) > 1:
                    diff = np.array(center) - np.array(best_target.center_list[-2])
                    best_target.diff_list.append(diff)
                else:
                    best_target.diff_list.append([0, 0])
                matched.append(best_target)
            else:
                new_target = Target(bbox=bbox, id=self.next_id, lost_frames=0, cls=0, age=1, conf_list=[1.0],
                                    diff_list=[[0, 0]], center_list=[center])
                self.targets.append(new_target)
                self.next_id += 1

        for target in self.targets:
            if target not in matched:
                target.lost_frames += 1
                target.conf_list.append(0.0)
                target.diff_list.append([0, 0])

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
                center_y - box_h / 2,
                center_x + box_w / 2,
                center_y + box_h / 2
            ]

            best_iou = 0
            best_target = None
            for target in self.targets:
                iou = self._calculate_iou(bbox, target.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target

            if best_iou > self.iou_threshold:
                best_target.bbox = bbox
                best_target.lost_frames = 0
                best_target.age += 1
                best_target.conf_list.append(conf)

                diff = self._calculate_frame_difference(self.previous_frame, frame_gray, best_target.bbox, frame.shape)
                best_target.diff_list.append(diff)

                matched.append(best_target)
            else:
                new_target = Target(bbox=bbox, id=self.next_id, lost_frames=0, cls=cls, age=1, conf_list=[conf],
                                    diff_list=[0.0])
                self.targets.append(new_target)
                self.next_id += 1

        for target in self.targets:
            if target not in matched:
                target.lost_frames += 1
                target.conf_list.append(0.0)
                diff = self._calculate_frame_difference(self.previous_frame, frame_gray, target.bbox, frame.shape)
                target.diff_list.append(diff)

        self.targets = [t for t in self.targets if t.lost_frames <= self.max_lost_frames]
        self.previous_frame = frame_gray

    def _calculate_frame_difference(self, prev_frame, curr_frame, bbox, frame_shape):
        """
        Calculate the frame difference for a given bounding box.

        Args:
            prev_frame: The previous frame.
            curr_frame: The current frame.
            bbox: The bounding box in normalized coordinates.
            frame_shape: The shape of the frame.

        Returns:
            The mean absolute difference between the patches in the previous and current frames.
        """
        if prev_frame is None:
            return 0.0
        prev_patch = self._extract_patch(prev_frame, bbox, frame_shape)
        curr_patch = self._extract_patch(curr_frame, bbox, frame_shape)
        if prev_patch.size > 0 and curr_patch.size > 0:
            return np.mean(np.abs(curr_patch - prev_patch))
        else:
            return 0.0

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
