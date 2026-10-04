"""
Dedicated Background Video Processing Worker Thread (QThread).
Decouples heavy video decoding, inference, and rendering from the Qt GUI main thread.
Includes automatic downscaling for high-resolution 2K/4K videos while preserving
exact original pixel coordinates for the streaming API.
"""
import time
import cv2
import numpy as np
from typing import Optional, List, Dict
from qtpy.QtCore import QThread, Signal

from core.video_stream import VideoStreamReader
from core.detector import ObjectDetectorTracker
from core.target_manager import TargetManager
from utils.drawing import render_detections
from utils.helpers import FPSCounter

class VideoProcessingWorker(QThread):
    # Signals to GUI thread
    frame_ready = Signal(np.ndarray)                # Rendered frame for display
    metrics_updated = Signal(float, int, float)      # fps, active_tracks, latency_ms
    telemetry_updated = Signal(dict)                # telemetry dict for target
    stream_finished = Signal()                      # Video reached end
    error_occurred = Signal(str)

    def __init__(
        self,
        source: str,
        is_rtsp: bool,
        detector: ObjectDetectorTracker,
        target_manager: TargetManager,
        api_server=None,
        max_proc_width: int = 640,
        parent=None
    ):
        super().__init__(parent)
        self.source = source
        self.is_rtsp = is_rtsp
        self.detector = detector
        self.target_manager = target_manager
        self.api_server = api_server
        self.max_proc_width = max_proc_width

        self.stream_reader: Optional[VideoStreamReader] = None
        self.is_running = False
        self.is_paused = False
        self.fps_counter = FPSCounter()
        self.pulse_counter = 0

    def run(self):
        self.is_running = True
        self.stream_reader = VideoStreamReader(source=self.source, is_rtsp=self.is_rtsp)
        if not self.stream_reader.start():
            self.error_occurred.emit(f"Failed to open source: {self.source}")
            self.is_running = False
            return

        frame_interval = 1.0 / self.stream_reader.fps if not self.is_rtsp else 0.001

        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            loop_start = time.perf_counter()
            frame = self.stream_reader.get_frame()

            if frame is None:
                if not self.stream_reader.is_connected:
                    # Video ended
                    self.stream_finished.emit()
                    break
                time.sleep(0.01)
                continue

            orig_h, orig_w = frame.shape[:2]

            # 1. High-Res Optimization (2K/4K videos like 2592x1520):
            # Downscale frame for inference & GUI rendering using fast LINEAR interpolation
            if orig_w > self.max_proc_width:
                scale = self.max_proc_width / float(orig_w)
                proc_w = self.max_proc_width
                proc_h = int(orig_h * scale)
                proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
                proc_w, proc_h = orig_w, orig_h
                proc_frame = frame

            # 2. Run Detection & Tracking
            t0 = time.perf_counter()
            detections = self.detector.process_frame(proc_frame, use_tracker=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 3. Scale coordinates back to original video resolution for API & Target Manager
            scaled_detections = []
            for det in detections:
                det_copy = dict(det)
                if scale != 1.0:
                    x1, y1, x2, y2 = det["bbox"]
                    cx, cy = det["center"]
                    det_copy["bbox"] = [
                        round(x1 / scale, 1),
                        round(y1 / scale, 1),
                        round(x2 / scale, 1),
                        round(y2 / scale, 1)
                    ]
                cx, cy = det_copy["center"]
                det_copy["center_pixel"] = {"x": round(cx, 1), "y": round(cy, 1)}
                det_copy["center_normalized"] = {"x": round(cx / orig_w, 4), "y": round(cy / orig_h, 4)}
                det_copy["frame_dims"] = (orig_w, orig_h)
                scaled_detections.append(det_copy)

            # 4. Update Target Tracking & Pixel Center Telemetry
            telemetry = self.target_manager.update(scaled_detections)
            if telemetry:
                self.telemetry_updated.emit(telemetry)

            if self.api_server:
                self.api_server.update_detections({
                    "timestamp": round(time.time(), 3),
                    "frame_resolution": {"width": orig_w, "height": orig_h},
                    "objects_count": len(scaled_detections),
                    "detections": scaled_detections
                })

            # 5. Render detections onto proc_frame for display
            self.pulse_counter += 1
            # For rendering on the display frame, use proc_frame coordinates
            trajectories_proc = {}
            for tid, trail in self.target_manager.trajectories.items():
                if scale != 1.0:
                    trajectories_proc[tid] = [(p[0] * scale, p[1] * scale) for p in trail]
                else:
                    trajectories_proc[tid] = list(trail)

            rendered_frame = render_detections(
                frame=proc_frame,
                detections=detections,
                selected_track_id=self.target_manager.selected_track_id,
                trajectories=trajectories_proc,
                pulse_frame_count=self.pulse_counter
            )

            # 6. Emit rendered frame to GUI thread and Web stream
            self.frame_ready.emit(rendered_frame)
            if self.api_server:
                self.api_server.update_frame(rendered_frame)

            # 7. Update metrics
            fps = self.fps_counter.tick()
            self.metrics_updated.emit(fps, len(self.target_manager.trajectories), latency_ms)

            # Frame rate pacing for video files (avoid burning 100% CPU)
            if not self.is_rtsp:
                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if self.stream_reader:
            self.stream_reader.stop()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def stop(self):
        self.is_running = False
        self.wait(1000)
