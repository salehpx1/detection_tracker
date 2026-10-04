"""
Sidebar Controls Panel for Video Input, Class Toggles, Hardware Device, and Detection Settings.
Includes Raspberry Pi & Low-Power edge optimizations.
"""
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QCheckBox, QSlider, QComboBox, QFileDialog, QRadioButton, QButtonGroup
)
from utils.helpers import get_available_devices

class ControlsPanel(QWidget):
    # Signals
    source_changed = Signal(str, bool)  # source_path_or_url, is_rtsp
    play_pause_clicked = Signal()
    stop_clicked = Signal()
    classes_filter_changed = Signal(bool, bool)  # detect_person, detect_car
    device_changed = Signal(str)  # 'auto', 'cuda', 'cpu'
    model_changed = Signal(str)
    conf_changed = Signal(float)
    iou_changed = Signal(float)
    tracker_changed = Signal(str)
    imgsz_changed = Signal(int)
    interval_changed = Signal(int)
    low_power_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(360)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 1. Video Source Card
        source_box = QGroupBox("📥 Input Source")
        sb_layout = QVBoxLayout(source_box)
        sb_layout.setSpacing(8)

        # Source mode selection
        mode_layout = QHBoxLayout()
        self.rb_file = QRadioButton("Video File")
        self.rb_rtsp = QRadioButton("RTSP Stream")
        self.rb_file.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_file)
        self.mode_group.addButton(self.rb_rtsp)
        mode_layout.addWidget(self.rb_file)
        mode_layout.addWidget(self.rb_rtsp)
        sb_layout.addLayout(mode_layout)

        # File selection row
        self.file_row = QWidget()
        f_layout = QHBoxLayout(self.file_row)
        f_layout.setContentsMargins(0, 0, 0, 0)
        self.txt_file = QLineEdit()
        self.txt_file.setPlaceholderText("Select video file (.mp4, .avi, ...)")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_file)
        f_layout.addWidget(self.txt_file)
        f_layout.addWidget(self.btn_browse)
        sb_layout.addWidget(self.file_row)

        # RTSP input row (hidden initially)
        self.rtsp_row = QWidget()
        r_layout = QHBoxLayout(self.rtsp_row)
        r_layout.setContentsMargins(0, 0, 0, 0)
        self.txt_rtsp = QLineEdit()
        self.txt_rtsp.setPlaceholderText("rtsp://username:password@ip:554/stream")
        r_layout.addWidget(self.txt_rtsp)
        sb_layout.addWidget(self.rtsp_row)
        self.rtsp_row.setVisible(False)

        # Connect mode toggle
        self.rb_file.toggled.connect(self._toggle_source_mode)

        # Play / Pause / Stop buttons
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Start Stream")
        self.btn_play.setObjectName("primaryButton")
        self.btn_play.clicked.connect(self._on_start_clicked)
        
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("dangerButton")
        self.btn_stop.clicked.connect(self.stop_clicked.emit)

        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_stop)
        sb_layout.addLayout(btn_layout)

        layout.addWidget(source_box)

        # 2. Dynamic Class Selection Card (PERSON / CAR TOGGLE)
        class_box = QGroupBox("🎯 Detection Targets")
        cb_layout = QVBoxLayout(class_box)
        cb_layout.setSpacing(8)

        self.chk_person = QCheckBox("🧑 Detect Person")
        self.chk_person.setChecked(True)
        self.chk_person.setStyleSheet("color: #4ADE80; font-weight: 600; font-size: 13px;")
        self.chk_person.toggled.connect(self._on_classes_changed)

        self.chk_car = QCheckBox("🚗 Detect Car & Vehicles")
        self.chk_car.setChecked(True)
        self.chk_car.setStyleSheet("color: #38BDF8; font-weight: 600; font-size: 13px;")
        self.chk_car.toggled.connect(self._on_classes_changed)

        cb_layout.addWidget(self.chk_person)
        cb_layout.addWidget(self.chk_car)
        layout.addWidget(class_box)

        # 3. Model & Low-Power / Raspberry Pi Settings Card
        hw_box = QGroupBox("⚙ Hardware & Edge Optimization")
        hw_layout = QVBoxLayout(hw_box)
        hw_layout.setSpacing(7)

        # Raspberry Pi / Low-Power Mode Toggle Preset
        self.chk_low_power = QCheckBox("⚡ Raspberry Pi Mode (Low Power)")
        self.chk_low_power.setChecked(True)
        self.chk_low_power.setStyleSheet("color: #FBBF24; font-weight: 700; font-size: 12px;")
        self.chk_low_power.toggled.connect(self._on_low_power_toggled)
        hw_layout.addWidget(self.chk_low_power)

        # Device Selection (GPU / CPU)
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("Device:"))
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["Auto Detect", "GPU (CUDA)", "CPU"])
        devices, gpu_name = get_available_devices()
        if "cuda" in devices:
            self.cmb_device.setItemText(1, f"GPU (CUDA: {gpu_name[:12]}...)")
        self.cmb_device.currentTextChanged.connect(self._on_device_changed)
        dev_layout.addWidget(self.cmb_device)
        hw_layout.addLayout(dev_layout)

        # Model Selection (yolov8n first for speed)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["yolov8n.pt (Ultra Light - 6MB)", "yolov8s.pt (Balanced)", "yolov8m.pt (Heavy)"])
        self.cmb_model.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.cmb_model)
        hw_layout.addLayout(model_layout)

        # Inference Resolution (imgsz)
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Inference Res:"))
        self.cmb_res = QComboBox()
        self.cmb_res.addItems(["320px (Pi / 75% Less CPU)", "416px (Balanced)", "640px (Desktop High Acc)"])
        self.cmb_res.currentTextChanged.connect(self._on_res_changed)
        res_layout.addWidget(self.cmb_res)
        hw_layout.addLayout(res_layout)

        # Frame Skipping / Detection Interval
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Detect Rate:"))
        self.cmb_skip = QComboBox()
        self.cmb_skip.addItems(["Every 2 Frames (50% CPU cut)", "Every 3 Frames (66% CPU cut)", "Every Frame (Heavy)"])
        self.cmb_skip.currentTextChanged.connect(self._on_skip_changed)
        skip_layout.addWidget(self.cmb_skip)
        hw_layout.addLayout(skip_layout)

        # Tracker Selection
        tracker_layout = QHBoxLayout()
        tracker_layout.addWidget(QLabel("Tracker:"))
        self.cmb_tracker = QComboBox()
        self.cmb_tracker.addItems(["bytetrack (Lightweight)", "botsort (BoT-SORT)"])
        self.cmb_tracker.currentTextChanged.connect(self._on_tracker_changed)
        tracker_layout.addWidget(self.cmb_tracker)
        hw_layout.addLayout(tracker_layout)

        # Confidence Slider
        self.lbl_conf = QLabel("Confidence: 0.40")
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(40)
        self.slider_conf.valueChanged.connect(self._on_conf_changed)
        hw_layout.addWidget(self.lbl_conf)
        hw_layout.addWidget(self.slider_conf)

        layout.addWidget(hw_box)

        # 4. Performance Telemetry Card
        perf_box = QGroupBox("📊 Performance HUD")
        pf_layout = QVBoxLayout(perf_box)
        pf_layout.setSpacing(6)

        m1_layout = QHBoxLayout()
        self.lbl_fps_val = QLabel("0.0 FPS")
        self.lbl_fps_val.setObjectName("metricValue")
        self.lbl_tracks_val = QLabel("0 Active")
        self.lbl_tracks_val.setObjectName("metricValue")
        m1_layout.addWidget(QLabel("FPS:"))
        m1_layout.addWidget(self.lbl_fps_val)
        m1_layout.addWidget(QLabel("Tracks:"))
        m1_layout.addWidget(self.lbl_tracks_val)
        pf_layout.addLayout(m1_layout)

        m2_layout = QHBoxLayout()
        self.lbl_latency_val = QLabel("0 ms")
        self.lbl_latency_val.setObjectName("metricValue")
        self.lbl_device_badge = QLabel("CPU")
        self.lbl_device_badge.setStyleSheet("background-color: #2D3748; padding: 2px 6px; border-radius: 4px; font-weight: bold; color: #10B981;")
        m2_layout.addWidget(QLabel("Inference:"))
        m2_layout.addWidget(self.lbl_latency_val)
        m2_layout.addWidget(QLabel("Device:"))
        m2_layout.addWidget(self.lbl_device_badge)
        pf_layout.addLayout(m2_layout)

        layout.addWidget(perf_box)

        layout.addStretch()

    def _toggle_source_mode(self):
        is_file = self.rb_file.isChecked()
        self.file_row.setVisible(is_file)
        self.rtsp_row.setVisible(not is_file)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm *.flv);;All Files (*.*)"
        )
        if path:
            self.txt_file.setText(path)

    def _on_start_clicked(self):
        is_rtsp = self.rb_rtsp.isChecked()
        source = self.txt_rtsp.text().strip() if is_rtsp else self.txt_file.text().strip()
        if source:
            self.source_changed.emit(source, is_rtsp)

    def _on_classes_changed(self):
        p = self.chk_person.isChecked()
        c = self.chk_car.isChecked()
        self.classes_filter_changed.emit(p, c)

    def _on_low_power_toggled(self, enabled: bool):
        if enabled:
            self.cmb_model.setCurrentIndex(0)  # yolov8n
            self.cmb_res.setCurrentIndex(0)    # 320px
            self.cmb_skip.setCurrentIndex(0)   # Every 2 frames
        self.low_power_toggled.emit(enabled)

    def _on_res_changed(self, text: str):
        size = int(text.split("px")[0].strip())
        self.imgsz_changed.emit(size)

    def _on_skip_changed(self, text: str):
        if "Every 2" in text:
            self.interval_changed.emit(2)
        elif "Every 3" in text:
            self.interval_changed.emit(3)
        else:
            self.interval_changed.emit(1)

    def _on_device_changed(self, text: str):
        if "GPU" in text:
            self.device_changed.emit("cuda")
        elif "CPU" in text:
            self.device_changed.emit("cpu")
        else:
            self.device_changed.emit("auto")

    def _on_model_changed(self, text: str):
        model_name = text.split(" ")[0]
        self.model_changed.emit(model_name)

    def _on_tracker_changed(self, text: str):
        tracker = text.split(" ")[0]
        self.tracker_changed.emit(tracker)

    def _on_conf_changed(self, val: int):
        f = val / 100.0
        self.lbl_conf.setText(f"Confidence: {f:.2f}")
        self.conf_changed.emit(f)

    def update_metrics(self, fps: float, tracks_count: int, latency_ms: float, device_str: str):
        self.lbl_fps_val.setText(f"{fps:.1f} FPS")
        self.lbl_tracks_val.setText(f"{tracks_count}")
        self.lbl_latency_val.setText(f"{latency_ms:.1f} ms")
        self.lbl_device_badge.setText(device_str.upper())
