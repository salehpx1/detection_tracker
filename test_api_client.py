"""
Utility script to test and demonstrate consuming real-time pixel coordinates from the API.
Supports:
1. REST polling (Target or All Detections)
2. WebSocket real-time client (Target or All Detections)
3. Local Webhook Receiver server (to receive POST streams)
"""
import sys
import time
import json
import argparse
import requests

def run_rest_poll(mode="all", host="localhost", port=8888, interval=0.2):
    endpoint = "/api/detections/current" if mode == "all" else "/api/target/current"
    url = f"http://{host}:{port}{endpoint}"
    print(f"[REST] Polling from: {url} (interval={interval}s)")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            try:
                resp = requests.get(url, timeout=1.5)
                if resp.status_code == 200:
                    data = resp.json()
                    if mode == "all":
                        dets = data.get("detections", [])
                        count = data.get("objects_count", len(dets))
                        items = []
                        for d in dets:
                            p = d.get("center_pixel") or {}
                            cx = p.get("x", d.get("center", [0, 0])[0])
                            cy = p.get("y", d.get("center", [0, 0])[1])
                            items.append(f"#{d.get('track_id')} {d.get('class_name')} @ ({cx:.1f}, {cy:.1f}) px")
                        summary = " | ".join(items) if items else "No objects"
                        print(f"[{count} objects] {summary}")
                    else:
                        if data.get("target_locked"):
                            p = data.get("center_pixel", {})
                            print(f"[LOCKED #{data.get('track_id')} {data.get('class_name')}]: "
                                  f"Center = (X: {p.get('x'):.1f}, Y: {p.get('y'):.1f}) px | Conf: {data.get('confidence')}")
                        else:
                            print("[IDLE] No target locked.")
            except Exception as e:
                print(f"[Error] Connection failed: {e}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")

def run_websocket_client(mode="all", host="localhost", port=8888):
    import asyncio
    try:
        import websockets
    except ImportError:
        print("[Error] Package 'websockets' not installed. Run: pip install websockets")
        return

    endpoint = "/ws/detections" if mode == "all" else "/ws/target"
    ws_url = f"ws://{host}:{port}{endpoint}"

    async def ws_loop():
        print(f"[WebSocket] Connecting to: {ws_url}...")
        async with websockets.connect(ws_url) as ws:
            print("[WebSocket] Connected! Streaming real-time coordinates:\n")
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if mode == "all":
                    dets = data.get("detections", [])
                    items = []
                    for d in dets:
                        p = d.get("center_pixel") or {}
                        cx = p.get("x", d.get("center", [0, 0])[0])
                        cy = p.get("y", d.get("center", [0, 0])[1])
                        items.append(f"#{d.get('track_id')} {d.get('class_name')}: ({cx:.0f}, {cy:.0f}) px")
                    print(f"[{len(dets)} detected] {' | '.join(items) if items else 'None'}")
                else:
                    if data.get("target_locked"):
                        p = data.get("center_pixel", {})
                        print(f"[TARGET #{data.get('track_id')} {data.get('class_name')}]: "
                              f"({p.get('x'):.1f}, {p.get('y'):.1f}) px | Time: {data.get('timestamp')}")
                    else:
                        print("[TARGET] None locked.")

    asyncio.run(ws_loop())

def run_webhook_receiver(port=5000):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class WebhookHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                if "detections" in payload:
                    dets = payload.get("detections", [])
                    print(f"[WEBHOOK] Received {len(dets)} detections:")
                    for d in dets:
                        p = d.get("center_pixel", {})
                        print(f"   -> ID #{d.get('track_id')} ({d.get('class_name')}): Center=({p.get('x')}, {p.get('y')}) px")
                elif "center_pixel" in payload:
                    p = payload.get("center_pixel", {})
                    print(f"[WEBHOOK] Target #{payload.get('track_id')} ({payload.get('class_name')}): Center=({p.get('x')}, {p.get('y')}) px")
            except Exception:
                print(f"[WEBHOOK RAW] {post_data.decode('utf-8')[:100]}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, format, *args):
            return

    server_address = ("", port)
    httpd = HTTPServer(server_address, WebhookHandler)
    print(f"[Webhook Receiver] Listening on http://localhost:{port}/")
    print(f"Pass --webhook-url http://localhost:{port}/ to stream coordinates directly here.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nReceiver stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Client for Vision Tracker Coordinate Streaming API")
    parser.add_argument("mode", choices=["rest-all", "rest-target", "ws-all", "ws-target", "webhook-server"],
                        default="rest-all", nargs="?",
                        help="Mode of operation (default: rest-all)")
    parser.add_argument("--host", default="localhost", help="API server host IP (e.g. 192.168.1.194)")
    parser.add_argument("--port", type=int, default=8888, help="API server port")
    parser.add_argument("--interval", type=float, default=0.2, help="Polling interval in seconds")
    parser.add_argument("--receiver-port", type=int, default=5000, help="Port for local webhook receiver")
    args = parser.parse_args()

    if args.mode == "rest-all":
        run_rest_poll(mode="all", host=args.host, port=args.port, interval=args.interval)
    elif args.mode == "rest-target":
        run_rest_poll(mode="target", host=args.host, port=args.port, interval=args.interval)
    elif args.mode == "ws-all":
        run_websocket_client(mode="all", host=args.host, port=args.port)
    elif args.mode == "ws-target":
        run_websocket_client(mode="target", host=args.host, port=args.port)
    elif args.mode == "webhook-server":
        run_webhook_receiver(port=args.receiver_port)
