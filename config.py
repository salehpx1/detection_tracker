"""
Application Configuration and Constants
"""
from dataclasses import dataclass, field
from typing import List, Dict

# COCO Class IDs
COCO_PERSON_CLASS_ID = 0
COCO_CAR_CLASS_ID = 2
COCO_MOTORCYCLE_CLASS_ID = 3
COCO_BUS_CLASS_ID = 5
COCO_TRUCK_CLASS_ID = 7

# Class categories
DETECTION_CLASSES = {
    "person": [COCO_PERSON_CLASS_ID],
    "car": [COCO_CAR_CLASS_ID, COCO_MOTORCYCLE_CLASS_ID, COCO_BUS_CLASS_ID, COCO_TRUCK_CLASS_ID],
}

CLASS_NAME_MAP = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# UI Theme Color Palette (Modern Dark / Cyber-Indigo)
COLORS = {
    "background": "#0F1117",
    "surface": "#1A1D27",
    "surface_secondary": "#232736",
    "border": "#2E344A",
    "text": "#F3F4F6",
    "text_muted": "#9CA3AF",
    "primary": "#6366F1",       # Indigo Accent
    "primary_hover": "#4F46E5",
    "accent_cyan": "#06B6D4",   # Cyan
    "accent_green": "#10B981",  # Emerald Green
    "accent_red": "#EF4444",    # Crimson Red
    "accent_amber": "#F59E0B",  # Amber/Yellow
    "person_box": (34, 197, 94),     # BGR/RGB for Person
    "car_box": (6, 182, 212),        # BGR/RGB for Car/Vehicles
    "selected_target": (239, 68, 68) # Red/Glowing for active clicked target
}

@dataclass
class AppConfig:
    # Model
    model_name: str = "yolov8n.pt"  # Default to ultralight nano model (6MB, ~3M params)
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.45
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    half_precision: bool = True  # FP16 if CUDA is used
    
    # Low-Power / Raspberry Pi Optimizations
    low_power_mode: bool = True
    imgsz: int = 320           # 320x320 uses 75% less CPU than 640x640
    detect_interval: int = 2   # Detect every N frames (2 = 50% CPU cut)
    cpu_threads: int = 2       # Cap PyTorch CPU threads to prevent 100% core saturation
    
    # Tracker
    tracker_type: str = "bytetrack"  # bytetrack (super lightweight O(1) latency)
    track_history_len: int = 20
    
    # Active Classes Filter
    detect_person: bool = True
    detect_car: bool = True
    
    # Video / Stream
    input_source: str = ""
    rtsp_buffer_size: int = 1
    stream_fps_limit: int = 60
    
    # Streaming API
    api_server_host: str = "0.0.0.0"
    api_server_port: int = 8888
    external_webhook_url: str = ""
    webhook_enabled: bool = False
    webhook_interval_ms: int = 33  # ~30 Hz
