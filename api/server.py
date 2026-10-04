"""
Built-in FastAPI & WebSocket API Server for streaming active target coordinates.
Includes endpoints for both single locked target and all detected objects,
plus ultra-low latency MJPEG live video streaming and an interactive web dashboard.
"""
import asyncio
import json
import threading
from typing import Optional, Dict, List, Set, Callable
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision Tracker - Edge Web Dashboard</title>
    <style>
        :root {
            --bg: #0B0D13;
            --surface: #161922;
            --border: #2A3042;
            --accent: #6366F1;
            --accent-cyan: #06B6D4;
            --accent-green: #10B981;
            --accent-red: #EF4444;
            --text: #F3F4F6;
            --muted: #9CA3AF;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 16px; display: flex; flex-direction: column; min-height: 100vh; }
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 14px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
        h1 { font-size: 1.25rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
        .badge { background: #1E2333; color: var(--accent-cyan); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid var(--border); }
        .main-container { display: grid; grid-template-columns: 1fr 360px; gap: 16px; flex: 1; }
        .video-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; position: relative; }
        .video-container { position: relative; flex: 1; display: flex; align-items: center; justify-content: center; background: #000; overflow: hidden; cursor: crosshair; }
        #videoFeed { max-width: 100%; max-height: 75vh; object-fit: contain; display: block; user-select: none; }
        .video-tip { position: absolute; bottom: 12px; left: 16px; background: rgba(0,0,0,0.7); backdrop-filter: blur(4px); padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; color: var(--muted); pointer-events: none; border: 1px solid rgba(255,255,255,0.1); }
        .sidebar { display: flex; flex-direction: column; gap: 14px; }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
        .card-title { font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: var(--accent); margin-bottom: 10px; letter-spacing: 0.5px; }
        .status-pill { display: block; text-align: center; padding: 8px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; background: #1F2432; color: var(--muted); margin-bottom: 12px; }
        .status-pill.locked { background: #064E3B; color: #34D399; border: 1px solid #059669; }
        .data-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; }
        .data-label { color: var(--muted); }
        .data-val { font-weight: 600; color: #fff; font-family: monospace; }
        .highlight { color: var(--accent-cyan); font-weight: 700; }
        .btn { width: 100%; padding: 8px 12px; border-radius: 6px; border: none; font-weight: 600; font-size: 0.85rem; cursor: pointer; transition: 0.2s; margin-top: 8px; }
        .btn-danger { background: var(--accent-red); color: #fff; }
        .btn-danger:hover { background: #DC2626; }
        .btn-primary { background: var(--accent); color: #fff; }
        .btn-primary:hover { background: #4F46E5; }
        .controls-group { display: flex; gap: 8px; margin-top: 10px; }
        .toggle-btn { flex: 1; padding: 7px; border-radius: 6px; border: 1px solid var(--border); background: #1A1F2C; color: #fff; font-size: 0.8rem; font-weight: 600; cursor: pointer; text-align: center; }
        .toggle-btn.active { border-color: var(--accent-green); background: #064E3B; color: #6EE7B7; }
        pre { background: #0E1017; padding: 10px; border-radius: 6px; font-size: 0.72rem; color: #34D399; overflow-x: auto; max-height: 180px; border: 1px solid var(--border); }
        .tab-btn { background: none; border: none; color: var(--muted); font-weight: 600; font-size: 0.8rem; cursor: pointer; padding: 4px 8px; margin-right: 6px; border-bottom: 2px solid transparent; }
        .tab-btn.active { color: var(--accent-cyan); border-bottom-color: var(--accent-cyan); }
    </style>
</head>
<body>
    <header>
        <h1>🎯 Vision Tracker <span>Edge & API Streamer</span></h1>
        <div>
            <span class="badge" style="color: #10B981;">NCNN / GPU Acceleration</span>
            <span class="badge" style="color: #38BDF8;">REST & WebSocket Online</span>
        </div>
    </header>

    <div class="main-container">
        <!-- Video Canvas Card -->
        <div class="video-card">
            <div class="video-container" id="videoBox">
                <img id="videoFeed" src="/video_feed" alt="Live Stream">
                <div class="video-tip">💡 Click on any bounding box to lock target & stream pixel center</div>
            </div>
        </div>

        <!-- Sidebar Telemetry & Controls -->
        <div class="sidebar">
            <div class="card">
                <div class="card-title">Target Lock & Pixel Center</div>
                <div class="status-pill" id="targetStatusPill">NO TARGET LOCKED (CLICK A BOX)</div>
                <div class="data-row">
                    <span class="data-label">Track Target:</span>
                    <span class="data-val highlight" id="targetLabel">None</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Pixel Center:</span>
                    <span class="data-val highlight" id="pixelCenter">(X: --, Y: --)</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Normalized:</span>
                    <span class="data-val" id="normCenter">(Nx: --, Ny: --)</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Bounding Box:</span>
                    <span class="data-val" id="bboxVal">[--, --, --, --]</span>
                </div>
                <button class="btn btn-danger" onclick="unlockTarget()">✕ Unlock Target</button>
            </div>

            <div class="card">
                <div class="card-title">Detection Filters</div>
                <div class="controls-group">
                    <div class="toggle-btn active" id="btnPerson" onclick="toggleClass('person')">🧑 Person</div>
                    <div class="toggle-btn active" id="btnCar" onclick="toggleClass('car')">🚗 Car / Vehicle</div>
                </div>
            </div>

            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div class="card-title" style="margin: 0;">Live API Telemetry</div>
                    <div>
                        <button class="tab-btn active" id="tabTarget" onclick="switchTab('target')">Target</button>
                        <button class="tab-btn" id="tabAll" onclick="switchTab('all')">All Objects</button>
                    </div>
                </div>
                <pre id="jsonPayload">{\n  "status": "waiting for stream..."\n}</pre>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('videoFeed');
        let detectPerson = true;
        let detectCar = true;
        let activeTab = 'target';
        let latestTargetData = null;
        let latestAllData = null;

        // Click on video to select target
        video.addEventListener('click', function(e) {
            const rect = video.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            const normX = clickX / rect.width;
            const normY = clickY / rect.height;

            fetch('/api/target/click', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ norm_x: normX, norm_y: normY })
            });
        });

        function unlockTarget() {
            fetch('/api/target/deselect', { method: 'POST' });
        }

        function toggleClass(type) {
            if (type === 'person') detectPerson = !detectPerson;
            if (type === 'car') detectCar = !detectCar;

            document.getElementById('btnPerson').classList.toggle('active', detectPerson);
            document.getElementById('btnCar').classList.toggle('active', detectCar);

            fetch(`/api/classes?person=${detectPerson}&car=${detectCar}`, { method: 'POST' });
        }

        function switchTab(tab) {
            activeTab = tab;
            document.getElementById('tabTarget').classList.toggle('active', tab === 'target');
            document.getElementById('tabAll').classList.toggle('active', tab === 'all');
            renderJson();
        }

        function renderJson() {
            const el = document.getElementById('jsonPayload');
            if (activeTab === 'target') {
                el.textContent = JSON.stringify(latestTargetData || { target_locked: false }, null, 2);
            } else {
                el.textContent = JSON.stringify(latestAllData || { count: 0, detections: [] }, null, 2);
            }
        }

        // WebSockets for target telemetry & all detections
        const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsTarget = new WebSocket(`${wsProtocol}//${location.host}/ws/target`);
        const wsDetections = new WebSocket(`${wsProtocol}//${location.host}/ws/detections`);

        wsTarget.onmessage = function(event) {
            latestTargetData = JSON.parse(event.data);
            const data = latestTargetData;

            const pill = document.getElementById('targetStatusPill');
            if (data.target_locked) {
                pill.textContent = '● TARGET LOCKED & STREAMING';
                pill.className = 'status-pill locked';

                document.getElementById('targetLabel').textContent = `${data.class_name} (ID: #${data.track_id})`;
                const cp = data.center_pixel || {};
                document.getElementById('pixelCenter').textContent = `(X: ${Math.round(cp.x)}, Y: ${Math.round(cp.y)}) px`;
                const cn = data.center_normalized || {};
                document.getElementById('normCenter').textContent = `(${cn.x.toFixed(4)}, ${cn.y.toFixed(4)})`;
                const bb = data.bbox || [];
                document.getElementById('bboxVal').textContent = `[${Math.round(bb[0])}, ${Math.round(bb[1])}, ${Math.round(bb[2])}, ${Math.round(bb[3])}]`;
            } else {
                pill.textContent = 'NO TARGET LOCKED (CLICK A BOX)';
                pill.className = 'status-pill';
                document.getElementById('targetLabel').textContent = 'None';
                document.getElementById('pixelCenter').textContent = '(X: --, Y: --)';
                document.getElementById('normCenter').textContent = '(Nx: --, Ny: --)';
                document.getElementById('bboxVal').textContent = '[--, --, --, --]';
            }
            if (activeTab === 'target') renderJson();
        };

        wsDetections.onmessage = function(event) {
            latestAllData = JSON.parse(event.data);
            if (activeTab === 'all') renderJson();
        };
    </script>
</body>
</html>
"""

class TargetApiServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8888,
        on_click_callback: Optional[Callable[[float, float], None]] = None,
        on_classes_callback: Optional[Callable[[bool, bool], None]] = None,
        on_deselect_callback: Optional[Callable[[], None]] = None,
        on_select_id_callback: Optional[Callable[[int], None]] = None
    ):
        self.host = host
        self.port = port
        self.on_click_callback = on_click_callback
        self.on_classes_callback = on_classes_callback
        self.on_deselect_callback = on_deselect_callback
        self.on_select_id_callback = on_select_id_callback

        self.latest_telemetry: Optional[Dict] = None
        self.latest_detections: Optional[Dict] = None
        self.latest_frame_jpeg: Optional[bytes] = None
        self.frame_id: int = 0
        
        self.active_websockets_target: Set[WebSocket] = set()
        self.active_websockets_detections: Set[WebSocket] = set()

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.server: Optional[uvicorn.Server] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False

        self.app = FastAPI(
            title="Vision Tracking & Coordinates Streaming API",
            description="Real-time pixel coordinate streaming API and Web Dashboard for tracked objects",
            version="2.1.0"
        )
        self._setup_routes()

    def _setup_routes(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/", response_class=HTMLResponse)
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            """Interactive Web Dashboard for streaming and click-to-lock tracking."""
            return HTMLResponse(content=HTML_DASHBOARD)

        @self.app.get("/video_feed")
        async def video_feed():
            """Ultra-low latency MJPEG live video stream endpoint."""
            async def frame_generator():
                last_sent_id = -1
                while self.is_running:
                    if self.latest_frame_jpeg is not None and self.frame_id != last_sent_id:
                        last_sent_id = self.frame_id
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + self.latest_frame_jpeg + b"\r\n"
                        )
                    await asyncio.sleep(0.005)  # 5ms check -> instant delivery
            return StreamingResponse(
                frame_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        # ---------------- TARGET ENDPOINTS ----------------
        @self.app.get("/api/target/current")
        async def get_current_target():
            """Returns the current locked target's coordinates and tracking telemetry."""
            if self.latest_telemetry:
                return JSONResponse(content=self.latest_telemetry)
            return JSONResponse(
                status_code=200,
                content={
                    "target_locked": False,
                    "message": "No target currently locked. Click on a bounding box or select a track ID."
                }
            )

        @self.app.post("/api/target/click")
        async def target_click(request: Request):
            """Endpoint to trigger target selection via web click coordinates."""
            body = await request.json()
            norm_x = body.get("norm_x", 0.0)
            norm_y = body.get("norm_y", 0.0)

            if self.on_click_callback:
                self.on_click_callback(norm_x, norm_y)
                return {"status": "ok", "message": f"Processed click at norm ({norm_x:.3f}, {norm_y:.3f})"}
            return {"status": "ignored", "message": "No callback registered"}

        @self.app.post("/api/target/select")
        async def target_select_id(track_id: int):
            """Select a specific track ID programmatically."""
            if self.on_select_id_callback:
                self.on_select_id_callback(track_id)
                return {"status": "ok", "selected_track_id": track_id}
            return {"status": "ignored"}

        @self.app.post("/api/target/deselect")
        async def target_deselect():
            """Unlock currently selected target."""
            if self.on_deselect_callback:
                self.on_deselect_callback()
            self.latest_telemetry = None
            return {"status": "ok", "message": "Target deselected"}

        # ---------------- ALL DETECTIONS ENDPOINTS ----------------
        @self.app.get("/api/detections/current")
        async def get_current_detections():
            """Returns all currently tracked objects with their pixel center coordinates."""
            if self.latest_detections:
                return JSONResponse(content=self.latest_detections)
            return JSONResponse(
                status_code=200,
                content={
                    "timestamp": 0,
                    "objects_count": 0,
                    "detections": []
                }
            )

        @self.app.post("/api/classes")
        async def toggle_classes(person: bool = True, car: bool = True):
            """Toggle Person and Car detection filters remotely."""
            if self.on_classes_callback:
                self.on_classes_callback(person, car)
            return {"status": "ok", "detect_person": person, "detect_car": car}

        @self.app.get("/api/target/stream")
        async def sse_target_stream():
            """Server-Sent Events (SSE) stream of target pixel coordinates."""
            async def event_generator():
                last_ts = 0.0
                while self.is_running:
                    if self.latest_telemetry:
                        curr_ts = self.latest_telemetry.get("timestamp", 0)
                        if curr_ts != last_ts:
                            last_ts = curr_ts
                            data = json.dumps(self.latest_telemetry)
                            yield f"data: {data}\n\n"
                    await asyncio.sleep(0.01)
            return StreamingResponse(event_generator(), media_type="text/event-stream")

        @self.app.get("/api/detections/stream")
        async def sse_detections_stream():
            """Server-Sent Events (SSE) stream of ALL detected objects' pixel centers."""
            async def event_generator():
                last_ts = 0.0
                while self.is_running:
                    if self.latest_detections:
                        curr_ts = self.latest_detections.get("timestamp", 0)
                        if curr_ts != last_ts:
                            last_ts = curr_ts
                            data = json.dumps(self.latest_detections)
                            yield f"data: {data}\n\n"
                    await asyncio.sleep(0.01)
            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # ---------------- WEBSOCKETS ----------------
        @self.app.websocket("/ws/target")
        async def websocket_target(websocket: WebSocket):
            """WebSocket endpoint for real-time target coordinate streaming."""
            await websocket.accept()
            self.active_websockets_target.add(websocket)
            try:
                if self.latest_telemetry:
                    await websocket.send_text(json.dumps(self.latest_telemetry))
                while self.is_running:
                    data = await websocket.receive_text()
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                self.active_websockets_target.discard(websocket)

        @self.app.websocket("/ws/detections")
        async def websocket_detections(websocket: WebSocket):
            """WebSocket endpoint streaming ALL detected objects' pixel centers in real-time."""
            await websocket.accept()
            self.active_websockets_detections.add(websocket)
            try:
                if self.latest_detections:
                    await websocket.send_text(json.dumps(self.latest_detections))
                while self.is_running:
                    data = await websocket.receive_text()
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                self.active_websockets_detections.discard(websocket)

    def update_frame(self, frame_bgr: np.ndarray):
        """Encodes frame to JPEG for the live web stream."""
        if frame_bgr is not None:
            ret, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                self.latest_frame_jpeg = buf.tobytes()
                self.frame_id += 1

    def update_telemetry(self, telemetry: Optional[Dict]):
        """Called with updated locked target data."""
        self.latest_telemetry = telemetry
        if self.loop and self.loop.is_running() and self.active_websockets_target and telemetry:
            msg = json.dumps(telemetry)
            coro = self._broadcast_ws(self.active_websockets_target, msg)
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    def update_detections(self, detections_payload: Optional[Dict]):
        """Called with updated detections payload containing all tracked objects."""
        self.latest_detections = detections_payload
        if self.loop and self.loop.is_running() and self.active_websockets_detections and detections_payload:
            msg = json.dumps(detections_payload)
            coro = self._broadcast_ws(self.active_websockets_detections, msg)
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _broadcast_ws(self, ws_set: Set[WebSocket], message: str):
        disconnected = set()
        for ws in ws_set:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        ws_set.difference_update(disconnected)

    def start(self):
        """Start the API server in a background daemon thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"[API Server] Web Dashboard & Stream running at http://{self.host}:{self.port}")

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio"
        )
        self.server = uvicorn.Server(config)
        self.loop.run_until_complete(self.server.serve())

    def stop(self):
        """Stop the API server."""
        self.is_running = False
        if self.server:
            self.server.should_exit = True
