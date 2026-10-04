"""
Helper utilities for hardware detection, FPS calculation, and coordinate mapping.
"""
import time
from collections import deque
from typing import Tuple, Optional

def get_available_devices():
    """Detect available compute devices (CUDA / CPU)."""
    devices = ["cpu"]
    gpu_name = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            devices.insert(0, "cuda")
    except Exception:
        pass
    return devices, gpu_name

def resolve_device(preference: str = "auto") -> str:
    """Resolve compute device based on preference and hardware availability."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available() and preference in ("auto", "cuda"):
            return "cuda:0"
    except Exception:
        pass
    return "cpu"

class FPSCounter:
    """Calculates smoothed frames per second."""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.frame_times.append(dt)
        return self.fps

    @property
    def fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg_dt = sum(self.frame_times) / len(self.frame_times)
        return (1.0 / avg_dt) if avg_dt > 0 else 0.0

def map_canvas_to_frame(
    click_x: float,
    click_y: float,
    canvas_w: int,
    canvas_h: int,
    frame_w: int,
    frame_h: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Map mouse click coordinates on a letterboxed/fitted UI canvas
    back to original video frame coordinates (x, y).
    """
    if canvas_w <= 0 or canvas_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return None, None

    # Calculate aspect ratios
    canvas_aspect = canvas_w / canvas_h
    frame_aspect = frame_w / frame_h

    if frame_aspect > canvas_aspect:
        # Frame is wider than canvas -> horizontal fit with top/bottom bars
        render_w = canvas_w
        render_h = int(canvas_w / frame_aspect)
        offset_x = 0
        offset_y = (canvas_h - render_h) // 2
    else:
        # Frame is taller than canvas -> vertical fit with left/right bars
        render_h = canvas_h
        render_w = int(canvas_h * frame_aspect)
        offset_x = (canvas_w - render_w) // 2
        offset_y = 0

    # Check if click is inside the rendered video area
    if not (offset_x <= click_x <= offset_x + render_w and offset_y <= click_y <= offset_y + render_h):
        return None, None

    # Compute normalized and map to frame resolution
    norm_x = (click_x - offset_x) / render_w
    norm_y = (click_y - offset_y) / render_h

    fx = norm_x * frame_w
    fy = norm_y * frame_h
    return fx, fy
