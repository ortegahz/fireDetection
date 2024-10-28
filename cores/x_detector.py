import inspect
import logging
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from mmpretrain import ImageClassificationInferencer


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
    flow_consistency_list: list = field(default_factory=list)
    flow_vector_list: list = field(default_factory=list)  # Add this to store the sum of flow vectors
    image_patches: list = field(default_factory=list)  # Store image patches
    flow_patches: list = field(default_factory=list)  # Store flow patches
    diff_patches: list = field(default_factory=list)
    conf_cls_list: list = field(default_factory=list)
    conf_cls_rgb_list: list = field(default_factory=list)


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
        _config = '/home/manu/mnt/ST2000DM005-2U91/workspace/mmpretrain/configs/resnet/resnet18_8xb32_fire.py'
        _checkpoint = '/media/manu/ST2000DM005-2U91/fire/mmpre/models/resnet18_8xb32_fire_flow/epoch_100.pth'
        self.model_cls_flow = ImageClassificationInferencer(model=_config, pretrained=_checkpoint, device='cuda')
        _checkpoint = '/media/manu/ST2000DM005-2U91/fire/mmpre/models/resnet18_8xb32_fire_rgb_v0/epoch_100.pth'
        self.model_cls_rgb = ImageClassificationInferencer(model=_config, pretrained=_checkpoint, device='cuda')

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

    def _patch_infer_flow(self, flow_patch, diff_patch):
        if len(flow_patch) > 0 and len(diff_patch) > 0:
            deltax = flow_patch[..., 0]
            deltay = flow_patch[..., 1]
            combined_display = cv2.merge((deltax.astype(np.uint8), diff_patch, deltay.astype(np.uint8)))
            result = self.model_cls_flow(combined_display)[0]
            cls_conf = result['pred_scores'][1]
            return cls_conf
        return 0.0

    def _patch_infer_rgb(self, img_patch):
        result = self.model_cls_rgb(img_patch)[0]
        cls_conf = result['pred_scores'][1]
        return cls_conf

    def update(self, detections, frame):
        fgmask = self.fgbg.apply(frame)
        matched = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for det in detections:
            # Parsing detection details...
            cls, center_x, center_y, box_w, box_h, conf = (
                map(float, det.split()) if len(det.split()) == 6 else (*map(float, det.split()), 0.0)
            )

            bbox = [
                center_x - box_w / 2,
                center_y - box_h / 2,
                center_x + box_w / 2,
                center_y + box_h / 2
            ]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            best_iou = 0
            best_target = None

            for target in self.targets:
                iou = self._calculate_iou(bbox, target.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target

            if best_iou > self.iou_threshold:
                # Update existing target properties...
                best_target.bbox = bbox
                best_target.bbox_list.append(bbox)
                best_target.lost_frames = 0
                best_target.age += 1
                best_target.conf_list.append(conf)

                diff = self._calculate_frame_difference(
                    self.previous_frame, frame_gray, best_target.bbox, best_target.last_valid_bbox, frame.shape
                )
                best_target.diff_list.append(diff)

                # Extract difference patch and update last valid bbox
                diff_patch = self._calculate_diff_patch(self.previous_frame, frame_gray, bbox, frame.shape)
                best_target.diff_patches.append(diff_patch)

                best_target.last_valid_bbox = bbox

                mask_avg = self._calculate_mask_avg(fgmask, best_target.bbox, frame.shape)
                best_target.mask_avg_list.append(mask_avg)

                best_target.area_list.append(area)
                area_diff = abs(area - best_target.area_list[-2]) if len(best_target.area_list) > 1 else 0.0
                best_target.area_diff_list.append(area_diff)

                # Extract and store image patch
                image_patch = self._extract_patch(frame, bbox, frame.shape)
                best_target.image_patches.append(image_patch)

                # Calculate and store optical flow patch if previous frame is available
                flow_patch = np.array([])
                if self.previous_frame is not None:
                    flow_patch = self._calculate_optical_flow(self.previous_frame, frame_gray, bbox, frame.shape)
                    best_target.flow_patches.append(flow_patch)

                _cls_conf_flow = self._patch_infer_flow(flow_patch, diff_patch)
                best_target.conf_cls_list.append(_cls_conf_flow)
                _cls_conf_rgb = self._patch_infer_rgb(image_patch)
                best_target.conf_cls_rgb_list.append(_cls_conf_rgb)

                matched.append(best_target)
            else:
                # Add new target
                new_target = Target(
                    bbox=bbox,
                    last_valid_bbox=bbox,
                    bbox_list=[bbox],
                    id=self.next_id,
                    lost_frames=0,
                    cls=cls,
                    age=1,
                    conf_list=[conf],
                    conf_cls_list=[0.0],
                    conf_cls_rgb_list=[0.0],  # match flow logic
                    diff_list=[0.0],
                    center_list=[(center_x, center_y)],
                    area_list=[area],
                    area_diff_list=[0.0],
                    mask_avg_list=[self._calculate_mask_avg(fgmask, bbox, frame.shape)],
                    image_patches=[self._extract_patch(frame, bbox, frame.shape)],
                    flow_patches=[],
                    diff_patches=[]
                )
                self.targets.append(new_target)
                self.next_id += 1

        for target in self.targets:
            if target not in matched:
                target.lost_frames += 1
                target.conf_list.append(-1.0)
                target.conf_cls_list.append(-1.0)
                target.conf_cls_rgb_list.append(-1.0)
                target.diff_list.append(0.0)

                mask_avg = self._calculate_mask_avg(fgmask, target.bbox, frame.shape)
                target.mask_avg_list.append(mask_avg)
                target.bbox_list.append(target.bbox)

        self.targets = [t for t in self.targets if t.lost_frames <= self.max_lost_frames]
        self.previous_frame = frame_gray

    def _calculate_diff_patch(self, prev_frame, curr_frame, bbox, frame_shape):
        if prev_frame is None or curr_frame is None:
            return np.array([])

        prev_patch = self._extract_patch(prev_frame, bbox, frame_shape)
        curr_patch = self._extract_patch(curr_frame, bbox, frame_shape)

        if prev_patch.size == 0 or curr_patch.size == 0:
            return np.array([])

        return cv2.absdiff(curr_patch, prev_patch)

    def _calculate_optical_flow(self, prev_frame, curr_frame, bbox, frame_shape):
        """
        Calculate the optical flow between two frames for a given bounding box area.

        Args:
            prev_frame: Grayscale image of the previous frame.
            curr_frame: Grayscale image of the current frame.
            bbox: Bounding box coordinates [x1, y1, x2, y2].
            frame_shape: Shape of the frames (height, width).

        Returns:
            A patch of the optical flow corresponding to the bounding box.
        """
        # Compute the dense optical flow using the Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Extract the optical flow patch corresponding to the bounding box
        flow_patch = self._extract_patch(flow, bbox, frame_shape)
        return flow_patch

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

        original_width = x2 - x1
        original_height = y2 - y1

        _aug_scale = 0.5
        width_increase = original_width * _aug_scale
        height_increase = original_height * _aug_scale

        x1 = int(max(0, (x1 - width_increase) * width))
        y1 = int(max(0, (y1 - height_increase) * height))
        x2 = int(min(width, (x2 + width_increase) * width))
        y2 = int(min(height, (y2 + height_increase) * height))

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


class SmokeDetector(FireDetector):
    def __init__(self, args):
        super(SmokeDetector, self).__init__(args)

    def update(self, detections, frame):
        fgmask = self.fgbg.apply(frame)

        matched = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = None
        if self.previous_frame is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(self.previous_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        for det in detections:
            # Parse detection
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

            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            best_iou = 0
            best_target = None
            for target in self.targets:
                iou = self._calculate_iou(bbox, target.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target

            if best_iou > self.iou_threshold:
                best_target.bbox = bbox
                best_target.bbox_list.append(bbox)
                best_target.lost_frames = 0
                best_target.age += 1
                best_target.conf_list.append(conf)

                diff = self._calculate_frame_difference(
                    self.previous_frame, frame_gray, best_target.bbox, best_target.last_valid_bbox, frame.shape
                )
                best_target.diff_list.append(diff)

                # Update last valid bbox to the current bbox
                best_target.last_valid_bbox = bbox

                mask_avg = self._calculate_mask_avg(fgmask, best_target.bbox, frame.shape)
                best_target.mask_avg_list.append(mask_avg)

                # Update area_list and area_diff_list
                best_target.area_list.append(area)
                if len(best_target.area_list) > 1:
                    area_diff = abs(area - best_target.area_list[-2])
                    best_target.area_diff_list.append(area_diff)
                else:
                    best_target.area_diff_list.append(0.0)

                # Calculate optical flow consistency
                if flow is not None:
                    flow_consistency = self._calculate_optical_flow_consistency(flow, best_target.bbox)
                    best_target.flow_consistency_list.append(flow_consistency)

                    # Calculate sum vector of flow
                    flow_vector_sum = self._calculate_flow_vector_sum(flow, best_target.bbox)
                    best_target.flow_vector_list.append(flow_vector_sum)

                matched.append(best_target)
            else:
                new_target = Target(
                    bbox=bbox,
                    last_valid_bbox=bbox,
                    bbox_list=[bbox],
                    id=self.next_id,
                    lost_frames=0,
                    cls=cls,
                    age=1,
                    conf_list=[conf],
                    diff_list=[0.0],
                    center_list=[(center_x, center_y)],
                    area_list=[area],
                    area_diff_list=[0.0],
                    mask_avg_list=[self._calculate_mask_avg(fgmask, bbox, frame.shape)],
                    flow_consistency_list=[0.0],  # Initialize with zero
                    flow_vector_list=[[0.0, 0.0]]  # Initialize flow vector sum with zero
                )
                self.targets.append(new_target)
                self.next_id += 1

        for target in self.targets:
            if target not in matched:
                target.lost_frames += 1
                target.conf_list.append(0.0)
                target.diff_list.append(0.0)

                mask_avg = self._calculate_mask_avg(fgmask, target.bbox, frame.shape)
                target.mask_avg_list.append(mask_avg)
                target.bbox_list.append(target.bbox)

                # Append zero for unmatched targets
                target.flow_consistency_list.append(0.0)
                target.flow_vector_list.append([0.0, 0.0])

        # Filter out targets
        self.targets = [t for t in self.targets if t.lost_frames <= self.max_lost_frames]
        self.previous_frame = frame_gray

    def _calculate_optical_flow_consistency(self, flow, bbox, magnitude_threshold=1.0):
        # Convert normalized bbox to integer pixel coordinates based on frame dimensions
        height, width = flow.shape[:2]
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        # Ensure the coordinates are within the image boundaries
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        # Check if the coordinates form a valid patch
        if x1 >= x2 or y1 >= y2:
            return 0.0

        flow_patch = flow[y1:y2, x1:x2]

        # Ensure the flow_patch is valid
        if flow_patch.size == 0:
            return 0.0

        # Use the angle of the flow vectors to determine direction consistency
        mag, ang = cv2.cartToPolar(flow_patch[..., 0], flow_patch[..., 1])

        # Filter out flow vectors with magnitude below the threshold
        valid_indices = mag > magnitude_threshold
        ang_filtered = ang[valid_indices]

        # Ensure there are enough valid angles for consistency calculation
        if ang_filtered.size == 0:
            return 0.0

        # Calculate the mean of filtered angles
        ang_mean = np.mean(ang_filtered)

        # Normalize angle differences to the range [-π, π]
        ang_diff = (ang_filtered - ang_mean + np.pi) % (2 * np.pi) - np.pi

        # Calculate angular standard deviation
        ang_std = np.std(ang_diff)

        # Calculate consistency as the inverse of the angular standard deviation
        ang_consistency = 1.0 / (ang_std + 1e-5)  # Small epsilon to prevent division by zero

        return ang_consistency

    def _calculate_flow_vector_sum(self, flow, bbox):
        height, width = flow.shape[:2]
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        if x1 >= x2 or y1 >= y2:
            return [0.0, 0.0]

        flow_patch = flow[y1:y2, x1:x2]

        if flow_patch.size == 0:
            return [0.0, 0.0]

        sum_flow_x = np.sum(flow_patch[..., 0])
        sum_flow_y = np.sum(flow_patch[..., 1])

        # Calculate the area of the bounding box
        bbox_area = (x2 - x1) * (y2 - y1)

        if bbox_area == 0:
            return [0.0, 0.0]

        # Normalize the flow vectors by the area of the bounding box
        normalized_flow_x = sum_flow_x / bbox_area
        normalized_flow_y = sum_flow_y / bbox_area

        return [normalized_flow_x * 64., normalized_flow_y * 64.]
