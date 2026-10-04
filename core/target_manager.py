"""
Target Selection and Pixel Center Tracking Manager.
"""
import time
from collections import deque
from typing import List, Dict, Optional, Tuple

class TargetManager:
    def __init__(self, history_len: int = 40):
        self.selected_track_id: Optional[int] = None
        self.selected_class_name: Optional[str] = None
        self.history_len = history_len
        self.trajectories: Dict[int, deque] = {}
        
        # Telemetry info
        self.last_telemetry: Optional[Dict] = None
        self.frames_lost: int = 0
        self.max_lost_frames: int = 30

    def select_target_by_click(
        self,
        click_x: float,
        click_y: float,
        detections: List[Dict]
    ) -> Optional[Dict]:
        """
        Finds the detection whose bounding box contains (click_x, click_y).
        Returns the matched detection or None.
        """
        matched = None
        min_area = float("inf")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                # If nested, pick the tighter box
                if area < min_area:
                    min_area = area
                    matched = det

        if matched:
            self.selected_track_id = matched.get("track_id", None)
            self.selected_class_name = matched.get("class_name", "Object")
            self.frames_lost = 0
            print(f"[TargetManager] Target selected: ID={self.selected_track_id} ({self.selected_class_name})")
            return matched
        else:
            print(f"[TargetManager] Click at ({click_x:.1f}, {click_y:.1f}) did not hit any box.")
            return None

    def deselect(self):
        """Clear active target selection."""
        self.selected_track_id = None
        self.selected_class_name = None
        self.last_telemetry = None
        self.frames_lost = 0

    def update(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Updates trajectories for all tracks and updates telemetry for the locked target.
        """
        # Update trajectories
        current_ids = set()
        active_target_det = None

        for det in detections:
            tid = det.get("track_id")
            cx, cy = det["center"]
            if tid is not None:
                current_ids.add(tid)
                if tid not in self.trajectories:
                    self.trajectories[tid] = deque(maxlen=self.history_len)
                self.trajectories[tid].append((cx, cy))

                if self.selected_track_id is not None and tid == self.selected_track_id:
                    active_target_det = det

        # Clean old trajectories
        dead_ids = [tid for tid in self.trajectories if tid not in current_ids and len(self.trajectories[tid]) > 0]
        for did in dead_ids:
            # Gradually shrink trail
            if len(self.trajectories[did]) > 0:
                self.trajectories[did].popleft()
            if len(self.trajectories[did]) == 0:
                del self.trajectories[did]

        # Process active target telemetry
        now = time.time()
        if self.selected_track_id is not None:
            if active_target_det is not None:
                self.frames_lost = 0
                fw, fh = active_target_det["frame_dims"]
                cx, cy = active_target_det["center"]
                norm_x, norm_y = active_target_det["center_norm"]

                self.last_telemetry = {
                    "target_locked": True,
                    "status": "TRACKING",
                    "track_id": self.selected_track_id,
                    "class_name": active_target_det["class_name"],
                    "confidence": active_target_det["confidence"],
                    "center_pixel": {
                        "x": cx,
                        "y": cy
                    },
                    "center_normalized": {
                        "x": norm_x,
                        "y": norm_y
                    },
                    "bbox": active_target_det["bbox"],
                    "frame_resolution": {
                        "width": fw,
                        "height": fh
                    },
                    "timestamp": round(now, 3)
                }
            else:
                self.frames_lost += 1
                if self.frames_lost < self.max_lost_frames and self.last_telemetry:
                    self.last_telemetry["status"] = "SEARCHING"
                    self.last_telemetry["target_locked"] = True
                else:
                    self.last_telemetry = {
                        "target_locked": False,
                        "status": "TARGET_LOST",
                        "track_id": self.selected_track_id,
                        "class_name": self.selected_class_name or "Unknown",
                        "timestamp": round(now, 3)
                    }
        else:
            self.last_telemetry = None

        return self.last_telemetry

    def get_trajectories_dict(self) -> Dict[int, List[Tuple[float, float]]]:
        """Convert deque trajectories to lists for rendering."""
        return {tid: list(trail) for tid, trail in self.trajectories.items()}
