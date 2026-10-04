# 🎯 Vision Tracker & Real-Time Pixel Coordinate Streamer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack-brightgreen.svg)](https://github.com/ifzhang/ByteTrack)
[![API](https://img.shields.io/badge/API-FastAPI%20%7C%20WebSockets%20%7C%20SSE-orange.svg)](https://fastapi.tiangolo.com/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Raspberry%20Pi%205-purple.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)

An industrial-grade, high-performance computer vision system for **real-time Person & Vehicle detection, multi-object tracking (MOT), and live pixel coordinate streaming** via REST API, WebSockets, Server-Sent Events (SSE), and Webhooks.

Built with **YOLOv8**, **ByteTrack**, **FastAPI**, and **PySide6**, supporting both a **rich Cyber-Dark Desktop GUI** and a **zero-dependency Headless/CLI Mode** optimized for edge computers such as the **Raspberry Pi 5** and cloud servers.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
  - [1. Headless / Pure CLI Mode (No GUI)](#1-headless--pure-cli-mode-no-gui-recommended-for-edge--servers)
  - [2. Desktop GUI Mode](#2-desktop-gui-mode)
  - [CLI Command-Line Options](#cli-command-line-options)
- [Real-Time API Documentation](#-real-time-api-documentation)
  - [REST Endpoints](#rest-endpoints)
  - [WebSocket Endpoints](#websocket-endpoints)
  - [Server-Sent Events (SSE)](#server-sent-events-sse)
  - [Live MJPEG Video Feed & Web Dashboard](#live-mjpeg-video-feed--web-dashboard)
  - [External Webhook Push Mode](#external-webhook-push-mode)
- [Testing & API Client Utility](#-testing--api-client-utility)
- [Raspberry Pi 5 & Edge Device Optimization](#-raspberry-pi-5--edge-device-optimization)
- [Directory Structure](#-directory-structure)

---

## 🌟 Key Features

1. **High-Accuracy Detection (Person & Vehicles):**
   - Powered by YOLOv8 models (`yolov8n.pt`, `yolov8s.pt`, etc.).
   - Dynamically toggle between **Person Only**, **Vehicles Only (Car, Bus, Truck, Motorcycle)**, or both simultaneously without interrupting the stream.

2. **Lightweight & Stable Multi-Object Tracking:**
   - Powered by **ByteTrack** for high track ID persistence even through occlusions.
   - Overlays the **Track ID** and **Class Name** (`#<id> <class>`) directly above each bounding box.
   - Historical motion trail visualization (trajectories).

3. **Two Operating Modes:**
   - **🖥️ Desktop GUI Mode:** Interactive dark HUD built with PySide6. Click on any bounding box to lock target and highlight its exact pixel center.
   - **⚡ Pure Headless / CLI Mode (`--no-gui`):** Zero Qt/GUI dependencies. Runs cleanly over SSH, in Docker containers, and on headless edge boards (Raspberry Pi 5). Automatically detects headless environments when `DISPLAY` is absent.

4. **Multi-Protocol Pixel Coordinate Streaming:**
   - Streams exact pixel centers `(x, y)` and normalized coordinates `(norm_x, norm_y)` for both:
     - **Active Locked Target** (via click or auto-lock).
     - **ALL Detected Objects** in the current frame simultaneously.
   - Protocols supported:
     - **REST API** (`/api/target/current`, `/api/detections/current`)
     - **WebSockets** (`/ws/target`, `/ws/detections`)
     - **Server-Sent Events (SSE)** (`/api/target/stream`, `/api/detections/stream`)
     - **Webhook Pusher** (asynchronously POSTs telemetry to your external endpoint at 30-60 Hz).

5. **Universal Video Input:**
   - Plays local video files (`.mp4`, `.avi`, `.mkv`, `.mov`, etc.).
   - Connects to **RTSP IP Camera Streams** with zero-latency buffer dropping and automatic reconnection.

6. **Hardware Acceleration:**
   - **NVIDIA GPU (CUDA):** Automatic detection with FP16 inference for 100+ FPS on RTX GPUs.
   - **Raspberry Pi 5 / ARM CPU:** Auto-detects and loads **NCNN** accelerated models (`yolov8n_ncnn_model`), achieving **~48 FPS (20.9 ms latency)** on a Raspberry Pi 5.
   - Bilinear fast downscaling for 2K/4K high-resolution camera feeds.

---

## 🏗 System Architecture

```
                                  +-----------------------+
                                  | Video File / RTSP Cam |
                                  +-----------+-----------+
                                              |
                                              v
                              +-------------------------------+
                              |    VideoStreamReader (Core)   |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |   YOLOv8 + ByteTrack Engine   |
                              |   (CUDA / NCNN / PyTorch CPU) |
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
       +----------------------------+                    +----------------------------+
       |   Headless Vision Engine   |                    |    PySide6 GUI Display     |
       |  (Zero Qt Dependencies)    |                    |     (Interactive HUD)      |
       +-------------+--------------+                    +-------------+--------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |   FastAPI & WebSocket Server  |
                              |   (Port 8888 / Configurable)  |
                              +---------------+---------------+
                                              |
         +--------------------+---------------+-------------------+--------------------+
         |                    |                                   |                    |
         v                    v                                   v                    v
  [REST Endpoints]    [WebSocket Feeds]                    [MJPEG Stream]       [Webhook Pusher]
  - /api/detections   - /ws/detections                     - /video_feed        - External HTTP POST
  - /api/target       - /ws/target                         - /dashboard
```

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/detection_tracker.git
cd detection_tracker
```

### 2. Create a virtual environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Raspberry Pi 5
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **For NVIDIA GPU (CUDA) acceleration:**
> Ensure PyTorch with CUDA support is installed:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 💻 How to Run

### 1. Headless / Pure CLI Mode (No GUI) [Recommended for Edge & Servers]
Run the application completely in the background without opening any GUI window. All bounding box pixel centers are automatically detected, tracked, and streamed over the API:

```bash
# Stream coordinates from a video file:
python main.py --no-gui --source people_sample.mp4

# Stream coordinates from an RTSP camera:
python main.py --no-gui --source "rtsp://admin:password@192.168.1.100:554/stream1" --rtsp

# Detect ONLY cars on a specific port:
python main.py --no-gui --source cars_sample.mp4 --car-only --api-port 8888

# Automatically push coordinates to your external webhook server:
python main.py --no-gui --source people_sample.mp4 --webhook-url "http://192.168.1.50:5000/api/receive"
```

*Note: If you run `python main.py` over an SSH session or inside a headless Linux container without an X11/Wayland display, the program automatically detects this and starts in Headless Mode.*

---

### 2. Desktop GUI Mode
Launch the interactive Cyber-Dark HUD application:

```bash
python main.py
```
- Click **"Browse Video"** or select **"RTSP Stream"**.
- Click **"Start Pipeline"**.
- **Click on any bounding box** in the video canvas to lock onto that object. The target is highlighted with a tactical reticle, and its exact pixel center coordinates are streamed live.
- Use the **Person** and **Car** checkboxes to toggle detection filters in real time.

---

### CLI Command-Line Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source` | `str` | `""` | Path to video file (`.mp4`, `.avi`, ...) or RTSP URL |
| `--rtsp` | flag | `False` | Must be set if `--source` is an RTSP stream |
| `--no-gui`, `--headless` | flag | `False` | Run in pure CLI / headless edge mode (zero Qt imports) |
| `--auto-track` | flag | `True` | Automatically lock onto the first detected object |
| `--person-only` | flag | `False` | Filter detections to Person only |
| `--car-only` | flag | `False` | Filter detections to Vehicles only (cars, trucks, buses) |
| `--device` | `str` | `auto` | Compute device: `auto`, `cuda`, or `cpu` |
| `--model` | `str` | `yolov8n.pt` | Path or name of YOLO model (`yolov8n.pt`, `yolov8s.pt`) |
| `--imgsz` | `int` | `320` | Inference resolution (`320` for Raspberry Pi, `640` for GPU) |
| `--api-host` | `str` | `0.0.0.0` | Host IP address for the built-in API server |
| `--api-port` | `int` | `8888` | Port for the built-in API server |
| `--webhook-url` | `str` | `""` | External HTTP URL to POST real-time coordinates to |

---

## 📡 Real-Time API Documentation

When the application runs (either in Headless or GUI mode), the built-in FastAPI server starts on `http://<host>:<port>` (default: `http://localhost:8888`).

Interactive Swagger UI documentation is available at:
👉 **`http://localhost:8888/docs`**

---

### REST Endpoints

#### 1. Get ALL Current Detections & Pixel Centers
Returns the pixel center coordinates and bounding boxes of all objects tracked in the latest frame.

- **URL:** `GET /api/detections/current`
- **Response Example (`200 OK`):**
```json
{
  "timestamp": 1728045612.345,
  "frame_resolution": {
    "width": 1920,
    "height": 1080
  },
  "objects_count": 2,
  "detections": [
    {
      "track_id": 1,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.892,
      "bbox": [512.4, 210.1, 640.8, 620.5],
      "center": [576.6, 415.3],
      "center_pixel": {
        "x": 576.6,
        "y": 415.3
      },
      "center_normalized": {
        "x": 0.3003,
        "y": 0.3845
      },
      "frame_dims": [1920, 1080]
    },
    {
      "track_id": 3,
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.941,
      "bbox": [1120.0, 540.2, 1450.6, 810.0],
      "center": [1285.3, 675.1],
      "center_pixel": {
        "x": 1285.3,
        "y": 675.1
      },
      "center_normalized": {
        "x": 0.6694,
        "y": 0.6251
      },
      "frame_dims": [1920, 1080]
    }
  ]
}
```

```bash
# Test with curl:
curl http://localhost:8888/api/detections/current
```

---

#### 2. Get Active Locked Target Coordinates
Returns telemetry and exact pixel center coordinates for the currently locked target.

- **URL:** `GET /api/target/current`
- **Response Example when target is locked (`200 OK`):**
```json
{
  "target_locked": true,
  "status": "TRACKING",
  "track_id": 1,
  "class_name": "person",
  "confidence": 0.892,
  "center_pixel": {
    "x": 576.6,
    "y": 415.3
  },
  "center_normalized": {
    "x": 0.3003,
    "y": 0.3845
  },
  "bbox": [512.4, 210.1, 640.8, 620.5],
  "frame_resolution": {
    "width": 1920,
    "height": 1080
  },
  "timestamp": 1728045612.345
}
```
- **Response Example when no target is locked:**
```json
{
  "target_locked": false,
  "message": "No target currently locked. Click on a bounding box or select a track ID."
}
```

```bash
# Test with curl:
curl http://localhost:8888/api/target/current
```

---

#### 3. Select Target by Track ID
Lock onto a specific object programmatically by its `track_id`:
- **URL:** `POST /api/target/select?track_id=1`
- **Response:** `{"status": "ok", "selected_track_id": 1}`

```bash
curl -X POST "http://localhost:8888/api/target/select?track_id=1"
```

---

#### 4. Select Target by Normalized Click Coordinates
Simulate a click on the video stream to lock onto whichever bounding box contains the coordinate:
- **URL:** `POST /api/target/click`
- **Body (`application/json`):**
```json
{
  "norm_x": 0.30,
  "norm_y": 0.38
}
```

---

#### 5. Unlock / Deselect Target
- **URL:** `POST /api/target/deselect`
- **Response:** `{"status": "ok", "message": "Target deselected"}`

---

#### 6. Toggle Class Filters Remotely
Change the active detection filter on the fly:
- **URL:** `POST /api/classes?person=true&car=false`
- **Response:** `{"status": "ok", "detect_person": true, "detect_car": false}`

---

### WebSocket Endpoints

WebSockets provide zero-latency, push-based streaming of pixel coordinates directly to client applications, frontend dashboards, or robotics controllers.

#### 1. All Detections Stream
Streams coordinates of all tracked objects on every processed frame:
- **URL:** `ws://<host>:<port>/ws/detections`

#### 2. Locked Target Stream
Streams coordinates of the active target on every processed frame:
- **URL:** `ws://<host>:<port>/ws/target`

**Python WebSocket Client Example:**
```python
import asyncio
import json
import websockets

async def listen_coordinates():
    uri = "ws://localhost:8888/ws/detections"
    async with websockets.connect(uri) as ws:
        print("Connected to Vision Tracker WebSocket!")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            for det in data.get("detections", []):
                tid = det["track_id"]
                cls = det["class_name"]
                cx = det["center_pixel"]["x"]
                cy = det["center_pixel"]["y"]
                print(f"Track #{tid} [{cls}] Center: X={cx}, Y={cy}")

asyncio.run(listen_coordinates())
```

---

### Server-Sent Events (SSE)

For web applications that prefer standard HTTP streaming:
- **All Detections:** `GET /api/detections/stream`
- **Locked Target:** `GET /api/target/stream`

---

### Live MJPEG Video Feed & Web Dashboard

The application includes an ultra-low latency live MJPEG video server:
- **Direct Video Stream:** `http://<host>:<port>/video_feed`
  (Can be directly embedded into any HTML page with `<img src="http://...:8888/video_feed">`).
- **Interactive Web Dashboard:** `http://<host>:<port>/dashboard`
  (Allows users over the local network to view the stream, click on bounding boxes to lock targets, and inspect telemetry in real time from any mobile phone, tablet, or browser).

---

### External Webhook Push Mode

If your architecture uses a central server or webhook endpoint, the Vision Tracker can automatically push coordinates directly to your server via HTTP POST:

```bash
python main.py --no-gui --source people_sample.mp4 --webhook-url "http://your-server.com/api/ingest"
```
The tracker pushes a JSON payload at a steady rate without blocking frame processing.

---

## 🧪 Testing & API Client Utility

The project includes an all-in-one testing client: [`test_api_client.py`](file:///c:/Users/px/Desktop/fun_project/detection_tracker/test_api_client.py).

```bash
# 1. Poll ALL detected objects' pixel centers via REST:
python test_api_client.py --mode rest-all --port 8888

# 2. Poll only the locked target coordinates via REST:
python test_api_client.py --mode rest-target --port 8888

# 3. Stream ALL pixel centers in real-time via WebSocket:
python test_api_client.py --mode ws-all --port 8888

# 4. Stream locked target via WebSocket:
python test_api_client.py --mode ws-target --port 8888

# 5. Start a local dummy Webhook server to test webhook pushes:
python test_api_client.py --mode webhook-server --port 5000
```

---

## 🍓 Raspberry Pi 5 & Edge Device Optimization

To achieve real-time performance on edge devices like the Raspberry Pi 5:

1. **Use NCNN Model Export:**
   NCNN provides high CPU vector acceleration on ARM Cortex-A76 cores.
   Export your model once with Ultralytics:
   ```bash
   yolo export model=yolov8n.pt format=ncnn imgsz=320
   ```
   This generates the `yolov8n_ncnn_model/` directory. The detector will automatically recognize and load this model when running on CPU!

2. **Benchmark Results on Raspberry Pi 5:**
   | Backend | Model | Image Size | Latency | FPS |
   |---|---|---|---|---|
   | PyTorch CPU | `yolov8n.pt` | 320x320 | ~88 ms | ~11 FPS |
   | **NCNN Engine** | **`yolov8n_ncnn_model`** | **320x320** | **~20.9 ms** | **~48 FPS** |

3. **Fast Downscaling:**
   For high-resolution cameras (e.g., 2K / 4K), the pipeline automatically downscales frames to a maximum processing width of 640 using fast bilinear interpolation (`cv2.INTER_LINEAR`), keeping decoding overhead under 3 ms while preserving detection accuracy.

4. **Recommended Raspberry Pi 5 Command:**
   ```bash
   python3 main.py --no-gui --source people_sample.mp4 --imgsz 320
   ```

---

## 📁 Directory Structure

```
detection_tracker/
├── main.py                     # Main application entry point (GUI + Headless CLI)
├── config.py                   # Central configurations & parameters
├── requirements.txt            # Python dependencies
├── test_api_client.py          # API & WebSocket test client utility
├── README.md                   # Complete system documentation
├── yolov8n.pt                  # YOLOv8 Nano weights
├── yolov8s.pt                  # YOLOv8 Small weights
├── core/
│   ├── detector.py             # YOLOv8 detector + NCNN loader + ByteTrack
│   ├── target_manager.py       # Target lock-on logic & coordinate computation
│   ├── video_stream.py         # Threaded video file & low-latency RTSP reader
│   ├── video_worker.py         # Qt-based background video worker for GUI
│   └── headless_pipeline.py    # Pure Python headless pipeline (Zero Qt)
├── api/
│   ├── server.py               # FastAPI server, WebSockets, SSE & Web Dashboard
│   └── client.py               # Asynchronous HTTP Webhook streamer
├── gui/
│   ├── main_window.py          # PySide6 Main Application Window
│   ├── video_canvas.py         # OpenGL/Qt canvas with click-to-lock mapping
│   ├── controls_panel.py       # Class toggles, sliders & controls
│   ├── api_panel.py            # Real-time API telemetry monitor HUD
│   └── styles.py               # Modern Cyber-Dark CSS / QSS Theme
└── utils/
    ├── drawing.py              # Tactical reticle, bounding boxes & labels
    └── helpers.py              # Hardware detection, FPS counter & helpers
```

---

## 📄 License
This project is open-source under the MIT License. Feel free to use and adapt for commercial or research projects.
