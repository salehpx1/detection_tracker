"""
HTTP Webhook client for streaming coordinates to external user-defined API endpoints.
"""
import time
import json
import threading
import queue
import requests
from typing import Optional, Dict, Callable

class WebhookStreamer:
    def __init__(
        self,
        endpoint_url: str = "",
        interval_ms: int = 50,
        on_status_update: Optional[Callable[[bool, int, float], None]] = None
    ):
        self.endpoint_url = endpoint_url
        self.interval = interval_ms / 1000.0
        self.on_status_update = on_status_update
        self.queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.is_enabled = False
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Stats
        self.packets_sent = 0
        self.last_status_code = 0
        self.last_latency_ms = 0.0

    def start(self):
        """Start sender thread."""
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.is_enabled = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def set_endpoint(self, url: str):
        self.endpoint_url = url.strip()

    def set_enabled(self, enabled: bool):
        self.is_enabled = enabled

    def push(self, telemetry: Dict):
        """Enqueue telemetry packet to be sent asynchronously."""
        if not self.is_enabled or not self.endpoint_url:
            return
        # If queue full, drop oldest
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put(telemetry)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                data = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.is_enabled or not self.endpoint_url:
                continue

            t0 = time.perf_counter()
            success = False
            status_code = 0
            try:
                resp = self.session.post(self.endpoint_url, json=data, timeout=0.8)
                status_code = resp.status_code
                success = (200 <= status_code < 300)
                self.packets_sent += 1
            except Exception as e:
                status_code = 0
                success = False

            latency = (time.perf_counter() - t0) * 1000.0
            self.last_latency_ms = round(latency, 1)
            self.last_status_code = status_code

            if self.on_status_update:
                self.on_status_update(success, status_code, self.last_latency_ms)

            # Enforce minimum interval
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        self.is_enabled = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
