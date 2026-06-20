"""
Evidence Logger — GridLock AI
Generates annotated evidence images with severity-coded overlays and logs to SQLite.
"""

import sqlite3
import cv2
import os
from datetime import datetime
import numpy as np


# Severity classification for each violation type
VIOLATION_SEVERITY = {
    "No Helmet":           {"level": "HIGH",     "color": (0, 0, 255),     "priority": 3},
    "No Seatbelt":         {"level": "MEDIUM",   "color": (0, 165, 255),   "priority": 2},
    "Triple Riding":       {"level": "HIGH",     "color": (0, 0, 255),     "priority": 3},
    "Wrong-Side Driving":  {"level": "CRITICAL", "color": (0, 0, 200),     "priority": 4},
    "Stop-Line Violation": {"level": "MEDIUM",   "color": (0, 165, 255),   "priority": 2},
    "Red-Light Violation": {"level": "CRITICAL", "color": (0, 0, 200),     "priority": 4},
    "Illegal Parking":     {"level": "LOW",      "color": (0, 200, 255),   "priority": 1},
}

# Badge colors for severity levels
SEVERITY_BADGE = {
    "LOW":      (0, 200, 255),    # Yellow
    "MEDIUM":   (0, 165, 255),    # Orange
    "HIGH":     (0, 0, 255),      # Red
    "CRITICAL": (0, 0, 180),      # Dark Red
}


class EvidenceLogger:
    def __init__(self, db_path="violations.db", evidence_dir="evidence"):
        self.db_path = db_path
        self.evidence_dir = evidence_dir
        
        if not os.path.exists(self.evidence_dir):
            os.makedirs(self.evidence_dir)
            
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database with extended schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                plate_number TEXT,
                violation_type TEXT,
                severity TEXT DEFAULT 'MEDIUM',
                confidence_score REAL,
                image_path TEXT
            )
        ''')
        
        # Schema Migration: Add severity column if it doesn't exist in older databases
        cursor.execute("PRAGMA table_info(violations)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'severity' not in columns:
            try:
                cursor.execute("ALTER TABLE violations ADD COLUMN severity TEXT DEFAULT 'MEDIUM'")
            except sqlite3.OperationalError:
                pass # Column might have been added concurrently
                
        conn.commit()
        conn.close()

    def _draw_severity_badge(self, image: np.ndarray, x: int, y: int, severity: str):
        """Draws a colored severity badge on the image."""
        badge_color = SEVERITY_BADGE.get(severity, (128, 128, 128))
        text = f" {severity} "
        
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Badge background
        cv2.rectangle(image, (x, y - text_h - 8), (x + text_w + 4, y), badge_color, -1)
        # Badge text
        cv2.putText(image, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_confidence_bar(self, image: np.ndarray, x: int, y: int, confidence: float, width: int = 100):
        """Draws a visual confidence bar on the image."""
        bar_h = 8
        # Background (dark)
        cv2.rectangle(image, (x, y), (x + width, y + bar_h), (50, 50, 50), -1)
        # Fill (green to red based on confidence)
        fill_w = int(width * confidence)
        # Color gradient: green (high conf) to red (low conf)
        r = int(255 * (1 - confidence))
        g = int(255 * confidence)
        cv2.rectangle(image, (x, y), (x + fill_w, y + bar_h), (0, g, r), -1)
        # Border
        cv2.rectangle(image, (x, y), (x + width, y + bar_h), (200, 200, 200), 1)

    def _draw_header_overlay(self, image: np.ndarray, timestamp: str, violation_count: int):
        """Draws a semi-transparent header bar with system info."""
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # Header bar
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        image[:] = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # System name
        cv2.putText(image, "GRIDLOCK AI", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        # Timestamp
        cv2.putText(image, timestamp, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        # Violation count badge
        count_text = f"VIOLATIONS: {violation_count}"
        cv2.putText(image, count_text, (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    def log_violation(self, original_image: np.ndarray, bbox: list, violation_type: str, 
                      confidence: float, plate_number: str) -> str:
        """
        Creates annotated evidence image with:
          - Severity-coded bounding box
          - Severity badge
          - Confidence bar
          - Plate number label
          - Header overlay with timestamp
        
        Saves to disk and logs metadata to SQLite.
        Returns path to saved evidence image.
        """
        annotated_image = original_image.copy()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Get severity info
        severity_info = VIOLATION_SEVERITY.get(violation_type, {"level": "MEDIUM", "color": (0, 165, 255), "priority": 2})
        severity = severity_info["level"]
        box_color = severity_info["color"]
        
        x1, y1, x2, y2 = bbox
        
        # 1. Draw bounding box with severity color (thicker for higher severity)
        thickness = severity_info["priority"] + 1
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, thickness)
        
        # 2. Semi-transparent fill for the bbox area
        overlay = annotated_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        annotated_image = cv2.addWeighted(overlay, 0.15, annotated_image, 0.85, 0)
        
        # 3. Violation label with background
        label = f"{violation_type} ({confidence:.0%})"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1 - 35, 50)
        
        cv2.rectangle(annotated_image, (x1, label_y - label_h - 6), (x1 + label_w + 8, label_y + 4), box_color, -1)
        cv2.putText(annotated_image, label, (x1 + 4, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        # 4. Severity badge
        self._draw_severity_badge(annotated_image, x1, label_y - label_h - 8, severity)
        
        # 5. Confidence bar below the bounding box
        bar_y = min(y2 + 5, annotated_image.shape[0] - 15)
        bar_width = min(x2 - x1, 120)
        self._draw_confidence_bar(annotated_image, x1, bar_y, confidence, bar_width)
        
        # 6. Plate number
        plate_label = f"PLATE: {plate_number}"
        cv2.putText(annotated_image, plate_label, (x1, bar_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 7. Header overlay
        self._draw_header_overlay(annotated_image, timestamp, 1)
        
        # 8. Save image
        image_filename = f"violation_{file_timestamp}.jpg"
        image_path = os.path.join(self.evidence_dir, image_filename)
        cv2.imwrite(image_path, annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 9. Log to Database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO violations (timestamp, plate_number, violation_type, severity, confidence_score, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, plate_number, violation_type, severity, confidence, image_path))
        conn.commit()
        conn.close()
        
        return image_path
