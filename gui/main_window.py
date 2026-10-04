"""
Main Window for the Vision Tracking and Coordinate Streaming Application.
Fully asynchronous architecture using dedicated background processing QThread
for zero GUI latency and optimal performance on Raspberry Pi 5 & Edge devices.
"""
import time
import cv2
import numpy as np
from typing import Optional, List, Dict
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStatusBar, QMessageBox
)

from config import AppConfig
from core.detector import ObjectDetectorTracker
from core.target_manager import TargetManager
from core.video_worker import VideoProcessingWorker
from api.server import TargetApiServer
from api.client import WebhookStreamer
from gui.video_canvas import VideoCanvas
from gui.controls_panel import ControlsPanel
from gui.api_panel import ApiPanel

class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("Vision Tracker & Pixel Coordinate Streamer - Edge/Pi5 Edition")
        self.resize(1440, 850)

        # Core Components
        self.detector = ObjectDetectorTracker(self.config)
        self.target_manager = TargetManager(history_len=self.config.track_history_len)
        self.worker: Optional[VideoProcessingWorker] = None

        self.api_server = TargetApiServer(
            host=self.config.api_server_host,
            port=self.config.api_server_port,
            on_click_callback=self._on_web_click,
            on_classes_callback=self._on_classes_filter_changed,
            on_deselect_callback=self._on_deselect_target
        )
        self.webhook_streamer = WebhookStreamer(
            endpoint_url=self.config.external_webhook_url,
            interval_ms=self.config.webhook_interval_ms,
            on_status_update=self._on_webhook_status
        )

        # UI Initialization
        self._init_ui()
        self._connect_signals()

        # Start built-in API server
        self.api_server.start()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left: Controls Sidebar
        self.controls_panel = ControlsPanel(self)
        main_layout.addWidget(self.controls_panel)

        # Center: Video Canvas
        self.video_canvas = VideoCanvas(self)
        main_layout.addWidget(self.video_canvas, stretch=1)

        # Right: API & Telemetry Panel
        self.api_panel = ApiPanel(self)
        main_layout.addWidget(self.api_panel)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select video source and press Start Stream.")

    def _connect_signals(self):
        # Canvas click interaction
        self.video_canvas.clicked_at_frame.connect(self._on_canvas_clicked)

        # Controls panel signals
        self.controls_panel.source_changed.connect(self._start_stream)
        self.controls_panel.play_pause_clicked.connect(self._toggle_pause)
        self.controls_panel.stop_clicked.connect(self._stop_stream)
        self.controls_panel.classes_filter_changed.connect(self._on_classes_filter_changed)
        self.controls_panel.device_changed.connect(self._on_device_changed)
        self.controls_panel.model_changed.connect(self._on_model_changed)
        self.controls_panel.conf_changed.connect(self._on_conf_changed)
        self.controls_panel.iou_changed.connect(self._on_iou_changed)
        self.controls_panel.tracker_changed.connect(self._on_tracker_changed)
        self.controls_panel.imgsz_changed.connect(self._on_imgsz_changed)
        self.controls_panel.interval_changed.connect(self._on_interval_changed)
        self.controls_panel.low_power_toggled.connect(self._on_low_power_toggled)

        # API panel signals
        self.api_panel.deselect_clicked.connect(self._on_deselect_target)
        self.api_panel.webhook_settings_changed.connect(self._on_webhook_settings_changed)

    def _start_stream(self, source: str, is_rtsp: bool):
        """Starts video file or RTSP stream in background worker."""
        self._stop_stream()
        self.status_bar.showMessage(f"Connecting to source: {source}...")

        self.worker = VideoProcessingWorker(
            source=source,
            is_rtsp=is_rtsp,
            detector=self.detector,
            target_manager=self.target_manager,
            api_server=self.api_server,
            max_proc_width=960,  # Limits 2.5K/4K video RAM allocation
            parent=self
        )

        # Connect worker signals to UI
        self.worker.frame_ready.connect(self.video_canvas.update_frame)
        self.worker.metrics_updated.connect(self._on_metrics_updated)
        self.worker.telemetry_updated.connect(self._on_telemetry_updated)
        self.worker.stream_finished.connect(self._on_stream_finished)
        self.worker.error_occurred.connect(self._on_worker_error)

        self.worker.start()
        self.status_bar.showMessage(f"Streaming from: {source}")

    def _toggle_pause(self):
        if self.worker and self.worker.isRunning():
            self.worker.toggle_pause()
            msg = "Stream Paused." if self.worker.is_paused else "Stream Resumed."
            self.status_bar.showMessage(msg)

    def _stop_stream(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.video_canvas.update_frame(None)
        self.status_bar.showMessage("Stream stopped.")

    def _on_metrics_updated(self, fps: float, active_tracks: int, latency_ms: float):
        dev_str = "NCNN" if getattr(self.detector, 'is_ncnn', False) else ("CUDA" if "cuda" in self.detector.device else "CPU")
        self.controls_panel.update_metrics(fps, active_tracks, latency_ms, dev_str)

    def _on_telemetry_updated(self, telemetry: dict):
        self.api_server.update_telemetry(telemetry)
        if telemetry and telemetry.get("target_locked", False):
            self.webhook_streamer.push(telemetry)
        self.api_panel.update_target_telemetry(telemetry)

    def _on_stream_finished(self):
        self._stop_stream()
        self.status_bar.showMessage("Playback finished.")

    def _on_worker_error(self, err_msg: str):
        QMessageBox.critical(self, "Stream Error", err_msg)
        self._stop_stream()

    def _on_canvas_clicked(self, fx: float, fy: float):
        """User clicked on canvas -> Lock onto matching target."""
        matched = self.target_manager.select_target_by_click(fx, fy, self.detector.last_detections)
        if matched:
            self.status_bar.showMessage(
                f"Locked on Target ID #{matched.get('track_id')} ({matched.get('class_name')}) at ({int(fx)}, {int(fy)}) px"
            )
        else:
            self.status_bar.showMessage(f"No bounding box at ({int(fx)}, {int(fy)}) px.")

    def _on_web_click(self, norm_x: float, norm_y: float):
        """Remote Web dashboard clicked on video -> Select target."""
        if not self.detector.last_detections:
            return
        fw, fh = self.detector.last_detections[0]["frame_dims"]
        fx = norm_x * fw
        fy = norm_y * fh
        self._on_canvas_clicked(fx, fy)

    def _on_deselect_target(self):
        self.target_manager.deselect()
        self.api_server.update_telemetry(None)
        self.api_panel.update_target_telemetry(None)
        self.status_bar.showMessage("Target unlocked.")

    def _on_classes_filter_changed(self, detect_person: bool, detect_car: bool):
        self.detector.set_classes_filter(detect_person, detect_car)
        mode_str = []
        if detect_person: mode_str.append("Person")
        if detect_car: mode_str.append("Car/Vehicles")
        self.status_bar.showMessage(f"Active Targets: {', '.join(mode_str) if mode_str else 'None'}")

    def _on_device_changed(self, device_pref: str):
        self.detector.set_device(device_pref)
        self.status_bar.showMessage(f"Device switched: {self.detector.device}")

    def _on_model_changed(self, model_name: str):
        self.detector.load_model(model_name)
        self.status_bar.showMessage(f"Model loaded: {model_name}")

    def _on_tracker_changed(self, tracker: str):
        self.config.tracker_type = tracker
        self.status_bar.showMessage(f"Tracker: {tracker}")

    def _on_conf_changed(self, val: float):
        self.config.confidence_threshold = val

    def _on_iou_changed(self, val: float):
        self.config.iou_threshold = val

    def _on_imgsz_changed(self, imgsz: int):
        self.detector.set_imgsz(imgsz)
        self.status_bar.showMessage(f"Inference size: {imgsz}px")

    def _on_interval_changed(self, interval: int):
        self.detector.set_detect_interval(interval)
        self.status_bar.showMessage(f"Frame skip interval: every {interval} frames")

    def _on_low_power_toggled(self, enabled: bool):
        self.config.low_power_mode = enabled
        if enabled:
            self.detector.set_imgsz(320)
            self.detector.set_detect_interval(2)
            self.status_bar.showMessage("⚡ Raspberry Pi Mode Active")
        else:
            self.detector.set_detect_interval(1)
            self.status_bar.showMessage("Desktop Mode Active")

    def _on_webhook_settings_changed(self, enabled: bool, url: str):
        self.webhook_streamer.set_enabled(enabled)
        self.webhook_streamer.set_endpoint(url)
        if enabled:
            self.webhook_streamer.start()

    def _on_webhook_status(self, success: bool, status_code: int, latency_ms: float):
        self.api_panel.update_webhook_stats(
            success=success,
            status_code=status_code,
            latency_ms=latency_ms,
            count=self.webhook_streamer.packets_sent
        )

    def closeEvent(self, event):
        """Cleanup on exit."""
        self._stop_stream()
        self.api_server.stop()
        self.webhook_streamer.stop()
        super().closeEvent(event)
