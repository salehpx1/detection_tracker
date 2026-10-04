"""
Modern Dark Theme Stylesheet for PySide6 GUI.
"""

DARK_THEME_QSS = """
/* Global styling */
QWidget {
    background-color: #0F1117;
    color: #E2E8F0;
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

/* Tooltips */
QToolTip {
    background-color: #1E2330;
    color: #F8FAFC;
    border: 1px solid #3B4252;
    padding: 6px;
    border-radius: 6px;
}

/* Group boxes and Cards */
QGroupBox {
    background-color: #171A24;
    border: 1px solid #282E40;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: 600;
    font-size: 13px;
    color: #818CF8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    background-color: #171A24;
}

/* Buttons */
QPushButton {
    background-color: #232838;
    color: #F1F5F9;
    border: 1px solid #333C52;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #2F364C;
    border-color: #6366F1;
    color: #FFFFFF;
}

QPushButton:pressed {
    background-color: #4F46E5;
    border-color: #4338CA;
}

QPushButton:disabled {
    background-color: #151821;
    color: #4B5563;
    border-color: #1F2432;
}

/* Accent Buttons */
QPushButton#primaryButton {
    background-color: #4F46E5;
    border: 1px solid #6366F1;
    color: #FFFFFF;
}

QPushButton#primaryButton:hover {
    background-color: #4338CA;
    border-color: #818CF8;
}

QPushButton#accentCyanButton {
    background-color: #0891B2;
    border: 1px solid #06B6D4;
    color: #FFFFFF;
}

QPushButton#accentCyanButton:hover {
    background-color: #0E7490;
}

QPushButton#dangerButton {
    background-color: #DC2626;
    border: 1px solid #EF4444;
    color: #FFFFFF;
}

QPushButton#dangerButton:hover {
    background-color: #B91C1C;
}

/* Inputs & ComboBox */
QLineEdit, QComboBox {
    background-color: #12151D;
    color: #F8FAFC;
    border: 1px solid #2B3245;
    border-radius: 8px;
    padding: 7px 12px;
    selection-background-color: #6366F1;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #6366F1;
    background-color: #141824;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left-width: 0px;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}

QComboBox QAbstractItemView {
    background-color: #171A24;
    border: 1px solid #333C52;
    selection-background-color: #4F46E5;
    color: #F8FAFC;
    padding: 4px;
}

/* Checkboxes */
QCheckBox {
    color: #E2E8F0;
    font-weight: 500;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #3E4760;
    background-color: #141722;
}

QCheckBox::indicator:hover {
    border-color: #6366F1;
}

QCheckBox::indicator:checked {
    background-color: #6366F1;
    border-color: #818CF8;
}

/* Sliders */
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #232838;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #6366F1;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #6366F1;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #818CF8;
    border-color: #FFFFFF;
}

/* ScrollBars */
QScrollBar:vertical {
    border: none;
    background: #10131B;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #293145;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4F46E5;
}

/* Labels */
QLabel {
    color: #CBD5E1;
}

QLabel#titleLabel {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
}

QLabel#metricValue {
    font-size: 16px;
    font-weight: 700;
    color: #38BDF8;
}

QLabel#metricLabel {
    font-size: 11px;
    color: #94A3B8;
    font-weight: 600;
}
"""
