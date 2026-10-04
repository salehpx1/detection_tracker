"""
Visualization and rendering utilities for detection, tracking, and target lock-on.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

def draw_corner_rect(img, pt1, pt2, color, thickness=2, corner_length=15):
    """Draw rectangle with accented corner brackets for a sleek modern HUD look."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    
    # Base bounding box (semi-transparent or thin)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thickness - 1), lineType=cv2.LINE_AA)
    
    # Corner brackets (thicker accents)
    t = thickness + 1
    cl = min(corner_length, (x2 - x1) // 3, (y2 - y1) // 3)
    if cl > 2:
        # Top-left
        cv2.line(img, (x1, y1), (x1 + cl, y1), color, t, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl), color, t, lineType=cv2.LINE_AA)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - cl, y1), color, t, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl), color, t, lineType=cv2.LINE_AA)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + cl, y2), color, t, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl), color, t, lineType=cv2.LINE_AA)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - cl, y2), color, t, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl), color, t, lineType=cv2.LINE_AA)

def draw_badge(img, text, pt, bg_color, text_color=(255, 255, 255), font_scale=0.55, thickness=1, padding=5):
    """Draw a styled badge/pill label with background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(pt[0]), int(pt[1])
    
    box_x1 = x
    box_y1 = max(0, y - text_h - 2 * padding)
    box_x2 = x + text_w + 2 * padding
    box_y2 = box_y1 + text_h + 2 * padding
    
    # Draw badge background
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1, lineType=cv2.LINE_AA)
    # Border
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    
    # Draw text
    text_y = box_y2 - padding - baseline // 2
    cv2.putText(img, text, (box_x1 + padding, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    return box_y1, box_y2

def draw_target_reticle(img, center_x, center_y, color=(0, 0, 255), radius=18, pulse=0):
    """Draw animated lock-on tactical reticle around the selected target center."""
    cx, cy = int(center_x), int(center_y)
    r = int(radius + pulse)
    
    # Outer circle dashed / arcs
    cv2.circle(img, (cx, cy), r, color, 2, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, r // 4), color, -1, lineType=cv2.LINE_AA)
    
    # Crosshairs extending outside
    gap = r + 4
    length = 10
    cv2.line(img, (cx - gap - length, cy), (cx - gap, cy), color, 2, lineType=cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (cx + gap + length, cy), color, 2, lineType=cv2.LINE_AA)
    cv2.line(img, (cx, cy - gap - length), (cx, cy - gap), color, 2, lineType=cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, cy + gap + length), color, 2, lineType=cv2.LINE_AA)

def draw_trajectory(img, trail: List[Tuple[float, float]], color=(0, 255, 255)):
    """Draw smoothed historical motion path of a tracked target."""
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        p2 = (int(trail[i][0]), int(trail[i][1]))
        # Fade out older points
        alpha = (i / len(trail))
        thickness = max(1, int(3 * alpha))
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

def render_detections(
    frame: np.ndarray,
    detections: List[Dict],
    selected_track_id: Optional[int] = None,
    trajectories: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    pulse_frame_count: int = 0
) -> np.ndarray:
    """
    Renders all detections, track IDs, classes, center points, and selected target HUD.
    """
    out_frame = frame.copy()
    pulse = (pulse_frame_count % 15) // 3

    for det in detections:
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        track_id = det.get("track_id", None)
        cls_name = det.get("class_name", "Object")
        conf = det.get("confidence", 0.0)
        cx, cy = det["center"]

        is_selected = (selected_track_id is not None and track_id == selected_track_id)
        
        # Color coding
        if is_selected:
            box_color = (0, 70, 255)       # Glowing Red/Orange for target lock
            badge_bg = (20, 20, 200)
            corner_len = 24
            thick = 3
        elif cls_name.lower() == "person":
            box_color = (34, 197, 94)      # Emerald Green (BGR)
            badge_bg = (20, 120, 50)
            corner_len = 16
            thick = 2
        else:  # Car / Vehicle
            box_color = (212, 182, 6)      # Cyan / Teal (BGR)
            badge_bg = (150, 100, 0)
            corner_len = 16
            thick = 2

        # Draw trajectory if available
        if trajectories and track_id in trajectories:
            trail = trajectories[track_id]
            trail_color = (0, 165, 255) if is_selected else (200, 200, 200)
            draw_trajectory(out_frame, trail, trail_color)

        # Draw Corner Rect
        x1, y1, x2, y2 = bbox
        draw_corner_rect(out_frame, (x1, y1), (x2, y2), box_color, thickness=thick, corner_length=corner_len)

        # Draw Label above bbox: "ID: {track_id} | {class_name}"
        if track_id is not None:
            label_text = f"ID: {track_id} | {cls_name}"
        else:
            label_text = f"{cls_name} [{conf:.2f}]"

        draw_badge(out_frame, label_text, (x1, y1), badge_bg, text_color=(255, 255, 255), font_scale=0.52)

        # Draw Center Point Dot
        cv2.circle(out_frame, (int(cx), int(cy)), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Highlight Selected Target
        if is_selected:
            # Tactical reticle
            draw_target_reticle(out_frame, cx, cy, color=(0, 0, 255), radius=22, pulse=pulse)
            
            # Pixel center coordinate badge below bbox
            coord_text = f"STREAMING CENTER: ({int(cx)}, {int(cy)}) px"
            draw_badge(out_frame, coord_text, (x1, y2 + 20), (0, 0, 180), text_color=(255, 255, 255), font_scale=0.48)

    return out_frame
