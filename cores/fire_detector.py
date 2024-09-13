import inspect
import logging
import os
import sys

from dataclasses import dataclass, field


@dataclass
class Target:
    bbox: list = field(default_factory=list)
    id: int = 0
    lost_frames: int = 0
    cls: float = 0.0
    age: int = 0
    conf_list: list = field(default_factory=list)


class FireDetector:
    def __init__(self, args):
        self.opt = self._build_yolo(args)
        self.targets = []
        self.next_id = 0
        self.iou_threshold = 0.1
        self.max_lost_frames = 5

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

    def update(self, detections):
        """
        Update targets based on new detections.

        Args:
            detections: List of detected bounding boxes in the format [x1, y1, x2, y2].
        """
        matched = []
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
                iou = self.calculate_iou(bbox, target.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target

            if best_iou > self.iou_threshold:
                best_target.bbox = bbox
                best_target.lost_frames = 0
                best_target.age += 1
                best_target.conf_list.append(conf)
                matched.append(best_target)
            else:
                new_target = Target(bbox=bbox, id=self.next_id, lost_frames=0, cls=cls, age=1, conf_list=[conf])
                self.targets.append(new_target)
                self.next_id += 1

        for target in self.targets:
            if target not in matched:
                target.lost_frames += 1
                target.conf_list.append(0.0)

        self.targets = [t for t in self.targets if t.lost_frames <= self.max_lost_frames]

    @staticmethod
    def calculate_iou(boxA, boxB):
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
