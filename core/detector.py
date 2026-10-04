"""
YOLO Detection and Tracking Engine with dynamic class filtering, device management,
and Raspberry Pi / Edge low-power optimizations.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from config import DETECTION_CLASSES, CLASS_NAME_MAP, AppConfig
from utils.helpers import resolve_device

class ObjectDetectorTracker:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.device = resolve_device(config.device)
        self.active_classes: List[int] = []
        self.frame_idx: int = 0
        self.last_detections: List[Dict] = []
        
        # Limit CPU threads if on CPU to protect low-power devices
        self._apply_cpu_optimizations()
        self.update_active_classes()
        self.load_model(config.model_name)

    def _apply_cpu_optimizations(self):
        """Limit CPU thread pool to keep CPU load low and prevent overheating."""
        if "cpu" in self.device:
            try:
                import torch
                threads = max(1, min(self.config.cpu_threads, 4))
                torch.set_num_threads(threads)
                print(f"[Detector] Low-Power Mode: PyTorch CPU threads capped at {threads}")
            except Exception as e:
                pass

    def update_active_classes(self):
        """Update active classes list based on current config flags."""
        classes = []
        if self.config.detect_person:
            classes.extend(DETECTION_CLASSES["person"])
        if self.config.detect_car:
            classes.extend(DETECTION_CLASSES["car"])
        self.active_classes = list(set(classes))

    def load_model(self, model_name: str):
        """Load YOLO model (auto-detects high-speed NCNN model on ARM/Pi)."""
        from ultralytics import YOLO
        self.config.model_name = model_name
        self.device = resolve_device(self.config.device)
        self._apply_cpu_optimizations()

        base_name = os.path.splitext(model_name)[0]
        ncnn_dir = f"{base_name}_ncnn_model"
        if "cpu" in self.device and os.path.exists(ncnn_dir):
            print(f"[Detector] Using optimized ARM/Pi NCNN engine: '{ncnn_dir}'")
            actual_model = ncnn_dir
            self.is_ncnn = True
        else:
            actual_model = model_name
            self.is_ncnn = False

        print(f"[Detector] Loading model '{actual_model}' on device '{self.device}' (imgsz={self.config.imgsz})...")
        self.model = YOLO(actual_model)
        # Warmup
        try:
            sz = self.config.imgsz
            dummy = np.zeros((sz, sz, 3), dtype=np.uint8)
            self.model(dummy, imgsz=sz, verbose=False)
        except Exception as e:
            pass

    def set_device(self, device_pref: str):
        """Change compute device ('auto', 'cuda', 'cpu')."""
        self.config.device = device_pref
        self.device = resolve_device(device_pref)
        self._apply_cpu_optimizations()
        print(f"[Detector] Switched device to: {self.device}")

    def set_classes_filter(self, detect_person: bool, detect_car: bool):
        """Dynamically update detection class filters."""
        self.config.detect_person = detect_person
        self.config.detect_car = detect_car
        self.update_active_classes()

    def set_imgsz(self, imgsz: int):
        """Update input image inference size (e.g. 320 for Raspberry Pi, 640 for desktop)."""
        self.config.imgsz = imgsz
        print(f"[Detector] Inference size set to {imgsz}px")

    def set_detect_interval(self, interval: int):
        """Set frame skip detection stride (e.g., detect every 2 frames)."""
        self.config.detect_interval = max(1, interval)
        print(f"[Detector] Frame skip detection interval set to: every {self.config.detect_interval} frames")

    def process_frame(
        self,
        frame: np.ndarray,
        use_tracker: bool = True
    ) -> List[Dict]:
        """
        Run detection and tracking with frame-skipping optimizations for edge devices.
        """
        if self.model is None or frame is None or len(self.active_classes) == 0:
            return []

        self.frame_idx += 1
        h, w = frame.shape[:2]

        # Edge optimization: Frame skipping (if interval > 1, reuse last detections on skipped frames)
        if self.config.detect_interval > 1 and (self.frame_idx % self.config.detect_interval != 0):
            if self.last_detections:
                return self.last_detections

        # Configure tracker yaml
        tracker_arg = f"{self.config.tracker_type}.yaml" if use_tracker else None

        try:
            if use_tracker:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=tracker_arg,
                    classes=self.active_classes,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.imgsz,
                    device=self.device,
                    verbose=False
                )
            else:
                results = self.model.predict(
                    source=frame,
                    classes=self.active_classes,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.imgsz,
                    device=self.device,
                    verbose=False
                )
        except Exception as e:
            print(f"[Detector] Inference error: {e}")
            return self.last_detections

        detections = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int) if (boxes.id is not None) else [None] * len(boxes)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cls_id = cls_ids[i]
                    conf = float(confs[i])
                    track_id = int(track_ids[i]) if track_ids[i] is not None else None
                    
                    # Compute pixel center in original frame resolution
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)

                    # Class name
                    cls_name = CLASS_NAME_MAP.get(cls_id, self.model.names.get(cls_id, f"Class {cls_id}"))

                    detections.append({
                        "track_id": track_id,
                        "class_id": int(cls_id),
                        "class_name": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                        "center": (round(cx, 1), round(cy, 1)),
                        "center_norm": (round(cx / w, 4), round(cy / h, 4)),
                        "frame_dims": (w, h)
                    })

        self.last_detections = detections
        return detections
