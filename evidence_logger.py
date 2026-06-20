import sqlite3
import cv2
import os
from datetime import datetime
import numpy as np

class EvidenceLogger:
    def __init__(self, db_path="violations.db", evidence_dir="evidence"):
        self.db_path = db_path
        self.evidence_dir = evidence_dir
        
        # Ensure evidence directory exists
        if not os.path.exists(self.evidence_dir):
            os.makedirs(self.evidence_dir)
            
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database and creates the table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                plate_number TEXT,
                violation_type TEXT,
                confidence_score REAL,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_violation(self, original_image: np.ndarray, bbox: list, violation_type: str, 
                      confidence: float, plate_number: str) -> str:
        """
        Draws annotations on the image, saves it to disk, and logs metadata to SQLite.
        Returns the relative path to the saved image.
        """
        # Make a copy to avoid drawing on the original reference
        annotated_image = original_image.copy()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 1. Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 2. Draw text labels
        label = f"{violation_type} ({confidence:.2f})"
        cv2.putText(annotated_image, label, (x1, max(y1 - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(annotated_image, f"Plate: {plate_number}", (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated_image, timestamp, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 3. Save image
        image_filename = f"violation_{file_timestamp}.jpg"
        image_path = os.path.join(self.evidence_dir, image_filename)
        cv2.imwrite(image_path, annotated_image)
        
        # 4. Log to Database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO violations (timestamp, plate_number, violation_type, confidence_score, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, plate_number, violation_type, confidence, image_path))
        conn.commit()
        conn.close()
        
        return image_path
