"""
API Control and Live Target Coordinate Telemetry Panel.
"""
from typing import Optional, Dict
from qtpy.QtCore import Qt, Signal, QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QCheckBox, QTextEdit
)

class ApiPanel(QWidget):
    # Signals
    deselect_clicked = Signal()
    webhook_settings_changed = Signal(bool, str)  # enabled, url

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # 1. Active Target Telemetry Card
        target_box = QGroupBox("🎯 Locked Target")
        tb_layout = QVBoxLayout(target_box)
        tb_layout.setSpacing(8)

        # Status badge
        self.lbl_status_badge = QLabel("NO TARGET SELECTED (CLICK A BOX)")
        self.lbl_status_badge.setAlignment(Qt.AlignCenter)
        self.lbl_status_badge.setStyleSheet(
            "background-color: #2D3748; color: #94A3B8; font-weight: bold; border-radius: 6px; padding: 6px;"
        )
        tb_layout.addWidget(self.lbl_status_badge)

        # Target metadata grid
        self.lbl_target_info = QLabel("Class / Track ID: None")
        self.lbl_target_info.setStyleSheet("font-size: 13px; font-weight: 600; color: #F1F5F9;")
        tb_layout.addWidget(self.lbl_target_info)

        # Coordinate details
        self.lbl_pixel_center = QLabel("Pixel Center: (X: --, Y: --)")
        self.lbl_pixel_center.setStyleSheet("font-size: 14px; font-weight: 700; color: #38BDF8;")
        tb_layout.addWidget(self.lbl_pixel_center)

        self.lbl_norm_center = QLabel("Normalized: (Nx: --, Ny: --)")
        self.lbl_norm_center.setStyleSheet("font-size: 12px; color: #94A3B8;")
        tb_layout.addWidget(self.lbl_norm_center)

        self.lbl_bbox = QLabel("BBox [x1, y1, x2, y2]: [--, --, --, --]")
        self.lbl_bbox.setStyleSheet("font-size: 11px; color: #64748B;")
        tb_layout.addWidget(self.lbl_bbox)

        # Deselect button
        self.btn_deselect = QPushButton("✕ Unlock / Clear Target")
        self.btn_deselect.setObjectName("dangerButton")
        self.btn_deselect.clicked.connect(self.deselect_clicked.emit)
        tb_layout.addWidget(self.btn_deselect)

        layout.addWidget(target_box)

        # 2. Local Streaming API Server Card
        api_box = QGroupBox("🌐 Built-in API Server (Local Stream)")
        api_layout = QVBoxLayout(api_box)
        api_layout.setSpacing(8)

        lbl_api_desc = QLabel("Real-time stream available at:")
        lbl_api_desc.setStyleSheet("color: #94A3B8; font-size: 11px;")
        api_layout.addWidget(lbl_api_desc)

        self.lbl_api_url = QLabel("http://localhost:8888/api/target/current")
        self.lbl_api_url.setStyleSheet("color: #A5B4FC; font-weight: 600; font-size: 11px;")
        api_layout.addWidget(self.lbl_api_url)

        self.lbl_ws_url = QLabel("WebSocket: ws://localhost:8888/ws/target")
        self.lbl_ws_url.setStyleSheet("color: #6EE7B7; font-size: 11px;")
        api_layout.addWidget(self.lbl_ws_url)

        btn_open_docs = QPushButton("Open API Docs / Swagger")
        btn_open_docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("http://localhost:8888/docs")))
        api_layout.addWidget(btn_open_docs)

        layout.addWidget(api_box)

        # 3. External API / Webhook Streamer Card
        hook_box = QGroupBox("🚀 External API Streamer (Push to Remote)")
        hb_layout = QVBoxLayout(hook_box)
        hb_layout.setSpacing(8)

        self.chk_webhook = QCheckBox("Enable POST Stream to URL")
        self.chk_webhook.toggled.connect(self._on_webhook_changed)
        hb_layout.addWidget(self.chk_webhook)

        self.txt_webhook_url = QLineEdit()
        self.txt_webhook_url.setPlaceholderText("http://your-server:5000/api/target")
        self.txt_webhook_url.textChanged.connect(self._on_webhook_changed)
        hb_layout.addWidget(self.txt_webhook_url)

        # Webhook Telemetry status
        self.lbl_hook_stats = QLabel("Status: Idle | Packets: 0")
        self.lbl_hook_stats.setStyleSheet("font-size: 11px; color: #94A3B8;")
        hb_layout.addWidget(self.lbl_hook_stats)

        layout.addWidget(hook_box)

        # 4. Live JSON Telemetry Log Box
        log_box = QGroupBox("📋 Live Telemetry Payload")
        lb_layout = QVBoxLayout(log_box)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet(
            "background-color: #0A0C11; color: #34D399; font-family: 'Consolas', monospace; font-size: 10px; border-radius: 6px;"
        )
        self.txt_log.setMaximumHeight(140)
        lb_layout.addWidget(self.txt_log)
        layout.addWidget(log_box)

        layout.addStretch()

    def _on_webhook_changed(self):
        enabled = self.chk_webhook.isChecked()
        url = self.txt_webhook_url.text().strip()
        self.webhook_settings_changed.emit(enabled, url)

    def update_target_telemetry(self, telemetry: Optional[Dict]):
        """Update active target display and live payload."""
        import json

        if telemetry and telemetry.get("target_locked", False):
            tid = telemetry.get("track_id", "Unknown")
            cname = telemetry.get("class_name", "Object")
            conf = telemetry.get("confidence", 0.0)
            status = telemetry.get("status", "TRACKING")

            if status == "TRACKING":
                self.lbl_status_badge.setText("● TARGET LOCKED & STREAMING")
                self.lbl_status_badge.setStyleSheet(
                    "background-color: #065F46; color: #34D399; font-weight: bold; border-radius: 6px; padding: 6px;"
                )
            else:
                self.lbl_status_badge.setText("● SEARCHING / OCCLUDED")
                self.lbl_status_badge.setStyleSheet(
                    "background-color: #78350F; color: #FBBF24; font-weight: bold; border-radius: 6px; padding: 6px;"
                )

            self.lbl_target_info.setText(f"Target: {cname} | Track ID: #{tid} ({conf*100:.0f}%)")

            cp = telemetry.get("center_pixel", {})
            self.lbl_pixel_center.setText(f"Pixel Center: (X: {cp.get('x', 0):.1f}, Y: {cp.get('y', 0):.1f}) px")

            cn = telemetry.get("center_normalized", {})
            self.lbl_norm_center.setText(f"Normalized: (Nx: {cn.get('x', 0):.4f}, Ny: {cn.get('y', 0):.4f})")

            bbox = telemetry.get("bbox", [])
            if len(bbox) == 4:
                self.lbl_bbox.setText(f"BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

            # Update live json log
            self.txt_log.setText(json.dumps(telemetry, indent=2))
        else:
            self.lbl_status_badge.setText("NO TARGET SELECTED (CLICK A BOX)")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #1E2433; color: #64748B; font-weight: bold; border-radius: 6px; padding: 6px;"
            )
            self.lbl_target_info.setText("Class / Track ID: None")
            self.lbl_pixel_center.setText("Pixel Center: (X: --, Y: --)")
            self.lbl_norm_center.setText("Normalized: (Nx: --, Ny: --)")
            self.lbl_bbox.setText("BBox: [--, --, --, --]")
            self.txt_log.setText('{\n  "target_locked": false\n}')

    def update_webhook_stats(self, success: bool, status_code: int, latency_ms: float, count: int):
        color = "#10B981" if success else "#EF4444"
        self.lbl_hook_stats.setText(
            f"HTTP {status_code} | {latency_ms:.1f} ms | Sent: {count}"
        )
        self.lbl_hook_stats.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: 600;")
