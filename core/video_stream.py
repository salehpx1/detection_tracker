"""
High-performance video and RTSP stream reader.
For files: Sequential, stutter-free frame reading.
For RTSP: Zero-latency grabber thread that drops stale frames to maintain real-time sync.
"""
import time
import threading
import cv2
import numpy as np
from typing import Optional

class VideoStreamReader:
    def __init__(self, source: str, is_rtsp: bool = False):
        self.source = source
        self.is_rtsp = is_rtsp
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.total_frames = 0
        self.is_connected = False
        
        # For RTSP live buffering
        self._latest_rtsp_frame: Optional[np.ndarray] = None
        self._rtsp_lock = threading.Lock()

    def start(self) -> bool:
        """Initialize capture."""
        self.stop_event.clear()
        self.pause_event.clear()

        src = int(self.source) if self.source.isdigit() else self.source

        if self.is_rtsp:
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap or not self.cap.isOpened():
            print(f"[VideoStream] Error opening source: {self.source}")
            return False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if (fps and fps > 0 and fps < 120) else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_rtsp else 0
        self.is_connected = True

        # For RTSP only: run background grabber thread
        if self.is_rtsp:
            self.thread = threading.Thread(target=self._rtsp_worker, daemon=True)
            self.thread.start()

        return True

    def _rtsp_worker(self):
        """Continuously grab latest RTSP frames without buffering lag."""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.05)
                continue

            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                continue

            # Grab to flush buffer
            grabbed = self.cap.grab()
            if grabbed:
                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    with self._rtsp_lock:
                        self._latest_rtsp_frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """Fetch next frame (sequential for files, latest for RTSP)."""
        if self.pause_event.is_set() or not self.is_connected:
            return None

        if self.is_rtsp:
            with self._rtsp_lock:
                return self._latest_rtsp_frame
        else:
            if not self.cap:
                return None
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.is_connected = False
                return None
            return frame

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()

    def is_paused(self) -> bool:
        return self.pause_event.is_set()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.is_connected = False
