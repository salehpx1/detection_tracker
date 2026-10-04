"""
Vision Tracker & Pixel Coordinate Streamer - Main Application Entry Point.
Supports:
1. Desktop GUI Mode (PyQt / PySide) with interactive click-to-lock targeting.
2. Pure Headless / No-GUI Mode: Zero Qt dependencies, streams all pixel centers
   and telemetry directly over REST API, WebSockets, and Webhooks.
"""
import sys
import os
import time
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="High-Accuracy Person & Vehicle Detection, Tracking, and Real-Time Pixel Coordinate Streaming API"
    )
    # Input source
    parser.add_argument("--source", type=str, default="", help="Path to video file (.mp4, .avi, ...) or RTSP URL")
    parser.add_argument("--rtsp", action="store_true", help="Specify this flag if source is an RTSP stream")

    # Execution Mode
    parser.add_argument("--no-gui", "--headless", dest="headless", action="store_true",
                        help="Run without GUI (pure CLI / Headless edge mode with REST & WebSocket API)")
    parser.add_argument("--auto-track", action="store_true", default=True,
                        help="Automatically lock onto and stream coordinates of the first detected object (default: True)")
    parser.add_argument("--webhook-url", type=str, default="",
                        help="External HTTP URL to automatically POST real-time coordinates to")

    # Classes filter
    parser.add_argument("--person-only", action="store_true", help="Detect ONLY people")
    parser.add_argument("--car-only", action="store_true", help="Detect ONLY cars/vehicles")

    # Hardware & Model
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Compute device ('auto', 'cuda', or 'cpu')")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model path or name (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--imgsz", type=int, default=320, choices=[320, 416, 640],
                        help="Inference resolution (320 for Raspberry Pi/Edge, 640 for GPU)")

    # API Server
    parser.add_argument("--api-host", type=str, default="0.0.0.0", help="Host IP for built-in API server")
    parser.add_argument("--api-port", type=int, default=8888, help="Port for built-in API server (default: 8888)")

    return parser.parse_args()

def run_headless(args, config: AppConfig):
    """Pure Python Headless Edge Mode (No Qt required)."""
    from core.headless_pipeline import HeadlessVisionPipeline
    from api.server import TargetApiServer
    from api.client import WebhookStreamer

    # Determine input source
    source = args.source
    if not source:
        for candidate in ["vid1715242010.mp4", "people_sample.mp4", "cars_sample.mp4"]:
            if os.path.exists(candidate):
                source = candidate
                break

    if not source:
        print("[Error] No input video file or RTSP stream specified. Use --source <path_or_rtsp>")
        sys.exit(1)

    print("=" * 68)
    print("  VISION TRACKER - HEADLESS / NO-GUI STREAMING ENGINE")
    print(f"  Source       : {source} (RTSP: {args.rtsp})")
    print(f"  Model/Device : {config.model_name} on {config.device.upper()} (imgsz={config.imgsz})")
    print(f"  API Server   : http://{config.api_server_host}:{config.api_server_port}")
    print(f"   |-- Detections API : http://{config.api_server_host}:{config.api_server_port}/api/detections/current")
    print(f"   |-- Target API     : http://{config.api_server_host}:{config.api_server_port}/api/target/current")
    print(f"   |-- WebSocket All  : ws://{config.api_server_host}:{config.api_server_port}/ws/detections")
    print(f"   |-- WebSocket Trgt : ws://{config.api_server_host}:{config.api_server_port}/ws/target")
    print(f"   |-- MJPEG Video    : http://{config.api_server_host}:{config.api_server_port}/video_feed")
    print(f"   \\-- Web Dashboard  : http://{config.api_server_host}:{config.api_server_port}/dashboard")
    if args.webhook_url:
        print(f"  Webhook POST : {args.webhook_url}")
    print("=" * 68)

    # Initialize Webhook Streamer if URL provided
    webhook_streamer = None
    if args.webhook_url:
        webhook_streamer = WebhookStreamer(endpoint_url=args.webhook_url, interval_ms=config.webhook_interval_ms)
        webhook_streamer.start()

    # Callbacks for web dashboard interaction
    pipeline = None

    def on_web_click(norm_x: float, norm_y: float):
        if pipeline and pipeline.detector.last_detections:
            fw, fh = pipeline.detector.last_detections[0]["frame_dims"]
            fx = norm_x * fw
            fy = norm_y * fh
            pipeline.target_manager.select_target_by_click(fx, fy, pipeline.detector.last_detections)

    def on_classes_filter(detect_person: bool, detect_car: bool):
        if pipeline:
            pipeline.detector.set_classes_filter(detect_person, detect_car)

    def on_deselect():
        if pipeline:
            pipeline.target_manager.deselect()

    def on_select_id(track_id: int):
        if pipeline:
            pipeline.target_manager.selected_track_id = track_id

    # Start FastAPI / WebSocket Server
    api_server = TargetApiServer(
        host=config.api_server_host,
        port=config.api_server_port,
        on_click_callback=on_web_click,
        on_classes_callback=on_classes_filter,
        on_deselect_callback=on_deselect,
        on_select_id_callback=on_select_id
    )
    api_server.start()

    # Start Headless Processing Pipeline
    pipeline = HeadlessVisionPipeline(
        config=config,
        source=source,
        is_rtsp=args.rtsp,
        auto_track_first=args.auto_track,
        max_proc_width=640,
        api_server=api_server,
        webhook_streamer=webhook_streamer
    )
    pipeline.start()

    print("\n[READY] Headless engine is streaming! Press Ctrl+C to stop.\n")
    try:
        while pipeline.is_running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\nShutting down headless engine...")
    finally:
        pipeline.stop()
        if webhook_streamer:
            webhook_streamer.stop()
        api_server.stop()
        print("Engine stopped cleanly.")

def run_gui(args, config: AppConfig):
    """Desktop GUI Mode (PyQt / PySide)."""
    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import Qt
    from gui.main_window import MainWindow
    from gui.styles import DARK_THEME_QSS

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Vision Tracker & Coordinate Streamer")
    app.setStyleSheet(DARK_THEME_QSS)

    window = MainWindow(config)
    window.show()

    # Auto-start source if provided via CLI
    if args.source:
        window.controls_panel.txt_file.setText(args.source)
        if args.rtsp:
            window.controls_panel.rb_rtsp.setChecked(True)
            window.controls_panel.txt_rtsp.setText(args.source)
        window._start_stream(args.source, args.rtsp)

    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())

def main():
    args = parse_args()

    # Determine class filters
    detect_person = not args.car_only
    detect_car = not args.person_only

    config = AppConfig(
        model_name=args.model,
        device=args.device,
        imgsz=args.imgsz,
        detect_person=detect_person,
        detect_car=detect_car,
        api_server_host=args.api_host,
        api_server_port=args.api_port,
        external_webhook_url=args.webhook_url,
        webhook_enabled=bool(args.webhook_url)
    )

    # Auto-detect headless environment if DISPLAY/WAYLAND not set
    is_headless = args.headless or (
        sys.platform != "win32" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    )

    if is_headless:
        run_headless(args, config)
    else:
        run_gui(args, config)

if __name__ == "__main__":
    main()
