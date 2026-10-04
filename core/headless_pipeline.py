"""
Pure Python Headless Vision Tracking Pipeline.
Completely independent of Qt/GUI libraries for minimal headless deployment on Raspberry Pi,
Linux servers, Docker containers, and edge embedded devices.
"""
import time
import os
import threading
import cv2
import numpy as np
from typing import Optional, Callable

from config import AppConfig
from core.detector import ObjectDetectorTracker
from core.target_manager import TargetManager
from core.video_stream import VideoStreamReader
from api.server import TargetApiServer
from api.client import WebhookStreamer
from utils.drawing import render_detections
from utils.helpers import FPSCounter

class HeadlessVisionPipeline:
    def __init__(
        self,
        config: AppConfig,
        source: str,
        is_rtsp: bool = False,
        auto_track_first: bool = True,
        max_proc_width: int = 640,
        api_server: Optional[TargetApiServer] = None,
        webhook_streamer: Optional[WebhookStreamer] = None
    ):
        self.config = config
        self.source = source
        self.is_rtsp = is_rtsp
        self.auto_track_first = auto_track_first
        self.max_proc_width = max_proc_width

        self.detector = ObjectDetectorTracker(config)
        self.target_manager = TargetManager(history_len=config.track_history_len)
        self.stream_reader: Optional[VideoStreamReader] = None
        self.api_server = api_server
        self.webhook_streamer = webhook_streamer

        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.fps_counter = FPSCounter()
        self.pulse_counter = 0

    def start(self):
        """Start the headless pipeline thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        self.stream_reader = VideoStreamReader(source=self.source, is_rtsp=self.is_rtsp)
        if not self.stream_reader.start():
            print(f"[HeadlessPipeline] Failed to open video source: {self.source}")
            self.is_running = False
            return

        print(f"[HeadlessPipeline] Pipeline active on '{self.source}' ({self.stream_reader.width}x{self.stream_reader.height})")
        frame_interval = 1.0 / self.stream_reader.fps if not self.is_rtsp else 0.001
        last_log_t = 0.0

        while self.is_running:
            loop_start = time.perf_counter()
            frame = self.stream_reader.get_frame()

            if frame is None:
                if not self.stream_reader.is_connected:
                    print("[HeadlessPipeline] Video source stream ended.")
                    break
                time.sleep(0.01)
                continue

            orig_h, orig_w = frame.shape[:2]

            # Downscale frame for fast inference using INTER_LINEAR
            if orig_w > self.max_proc_width:
                scale = self.max_proc_width / float(orig_w)
                proc_w = self.max_proc_width
                proc_h = int(orig_h * scale)
                proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
                proc_w, proc_h = orig_w, orig_h
                proc_frame = frame

            # Run YOLO + ByteTrack
            t0 = time.perf_counter()
            detections = self.detector.process_frame(proc_frame, use_tracker=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Scale coordinates back to original resolution
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
                    det_copy["center"] = (round(cx / scale, 1), round(cy / scale, 1))
                cx, cy = det_copy["center"]
                det_copy["center_pixel"] = {"x": round(cx, 1), "y": round(cy, 1)}
                det_copy["center_normalized"] = {"x": round(cx / orig_w, 4), "y": round(cy / orig_h, 4)}
                det_copy["frame_dims"] = (orig_w, orig_h)
                scaled_detections.append(det_copy)

            now = time.time()

            # 1. Update ALL detections payload in API server
            all_detections_payload = {
                "timestamp": round(now, 3),
                "frame_resolution": {"width": orig_w, "height": orig_h},
                "objects_count": len(scaled_detections),
                "detections": scaled_detections
            }
            if self.api_server:
                self.api_server.update_detections(all_detections_payload)

            # 2. Target Management
            # If auto-track is on and no target currently locked, lock onto the first detected object
            if self.auto_track_first and self.target_manager.selected_track_id is None and len(scaled_detections) > 0:
                first_tid = scaled_detections[0].get("track_id")
                if first_tid is not None:
                    self.target_manager.selected_track_id = first_tid
                    self.target_manager.selected_class_name = scaled_detections[0].get("class_name")

            telemetry = self.target_manager.update(scaled_detections)
            if self.api_server:
                self.api_server.update_telemetry(telemetry)

            # Webhook push
            if self.webhook_streamer and self.webhook_streamer.is_enabled:
                payload_to_push = telemetry if telemetry else all_detections_payload
                self.webhook_streamer.push(payload_to_push)

            # 3. Render frame for MJPEG live web feed if API server is active
            if self.api_server:
                self.pulse_counter += 1
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
                self.api_server.update_frame(rendered_frame)

            # 4. Metrics & Console Output
            fps = self.fps_counter.tick()
            if time.time() - last_log_t > 0.5:
                last_log_t = time.time()
                obj_strs = [
                    f"#{d.get('track_id') or '?'} {d['class_name']}: ({int(d['center'][0])}, {int(d['center'][1])}) px"
                    for d in scaled_detections[:3]
                ]
                obj_summary = " | ".join(obj_strs) if obj_strs else "No objects"
                dev_str = "NCNN" if getattr(self.detector, 'is_ncnn', False) else self.detector.device.upper()
                print(f"\r[{dev_str} | FPS: {fps:4.1f} | {latency_ms:4.1f}ms] Detected: {len(scaled_detections)} objects -> {obj_summary}", end="", flush=True)

            # Frame rate pacing for video files
            if not self.is_rtsp:
                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if self.stream_reader:
            self.stream_reader.stop()
        self.is_running = False
        print("\n[HeadlessPipeline] Stopped.")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
