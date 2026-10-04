"""
High-performance interactive video rendering widget with mouse click mapping.
"""
import cv2
import numpy as np
from qtpy.QtCore import Qt, Signal, QRectF
from qtpy.QtGui import QPainter, QImage, QColor, QFont, QPen
from qtpy.QtWidgets import QWidget
from utils.helpers import map_canvas_to_frame

class VideoCanvas(QWidget):
    """
    Renders video frames and converts screen clicks to original video frame coordinates.
    """
    clicked_at_frame = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #0B0D13; border-radius: 12px;")
        self.setMouseTracking(True)
        
        self.current_frame: np.ndarray = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.last_click_point = None

    def update_frame(self, frame: np.ndarray):
        """Update and repaint with new BGR frame."""
        self.current_frame = frame
        if frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.frame_width > 0 and self.frame_height > 0:
            if hasattr(event, "position"):
                click_x = event.position().x()
                click_y = event.position().y()
            else:
                click_x = event.pos().x()
                click_y = event.pos().y()

            fx, fy = map_canvas_to_frame(
                click_x=click_x,
                click_y=click_y,
                canvas_w=self.width(),
                canvas_h=self.height(),
                frame_w=self.frame_width,
                frame_h=self.frame_height
            )
            if fx is not None and fy is not None:
                self.last_click_point = (click_x, click_y)
                self.clicked_at_frame.emit(fx, fy)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w = self.width()
        h = self.height()

        if self.current_frame is None or self.frame_width == 0 or self.frame_height == 0:
            # Render empty state placeholder
            painter.fillRect(0, 0, w, h, QColor("#0B0D13"))
            
            # Subtle grid or border
            painter.setPen(QPen(QColor("#1E2433"), 2, Qt.DashLine))
            painter.drawRoundedRect(10, 10, w - 20, h - 20, 10, 10)

            # Center text
            painter.setPen(QColor("#6366F1"))
            font_title = QFont("Segoe UI", 16, QFont.Bold)
            painter.setFont(font_title)
            painter.drawText(QRectF(0, h // 2 - 40, w, 35), Qt.AlignCenter, "TARGET DETECTION & TRACKING ENGINE")

            painter.setPen(QColor("#94A3B8"))
            font_sub = QFont("Segoe UI", 11)
            painter.setFont(font_sub)
            painter.drawText(
                QRectF(0, h // 2 + 5, w, 40),
                Qt.AlignCenter,
                "Select a video file or connect an RTSP stream to begin.\nClick on any bounding box to stream its pixel center."
            )
            return

        # Frame Aspect Ratio calculation
        frame_aspect = self.frame_width / self.frame_height
        canvas_aspect = w / h

        if frame_aspect > canvas_aspect:
            render_w = w
            render_h = int(w / frame_aspect)
            offset_x = 0
            offset_y = (h - render_h) // 2
        else:
            render_h = h
            render_w = int(h * frame_aspect)
            offset_x = (w - render_w) // 2
            offset_y = 0

        # Convert OpenCV BGR to RGB QImage
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * self.frame_width
        q_img = QImage(rgb_frame.data, self.frame_width, self.frame_height, bytes_per_line, QImage.Format_RGB888)

        # Clear background bars
        painter.fillRect(0, 0, w, h, QColor("#0B0D13"))

        # Draw frame scaled smoothly
        target_rect = QRectF(offset_x, offset_y, render_w, render_h)
        painter.drawImage(target_rect, q_img)
