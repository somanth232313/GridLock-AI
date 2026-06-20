"""
ANPR Engine — GridLock AI
Automatic Number Plate Recognition using contour-based plate localization + EasyOCR.

Pipeline:
  Vehicle Crop → Grayscale → Bilateral Filter → Adaptive Threshold → 
  Contour Detection → Plate Candidate Filtering → Perspective Correction → OCR
"""

import cv2
import numpy as np
import easyocr
import re

from utils import safe_crop


class PlateReader:
    def __init__(self):
        """
        Initialize EasyOCR reader.
        Set gpu=True if CUDA is available for faster processing.
        """
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Indian license plate regex pattern (e.g., MH12AB1234, KA01HH1234)
        self.plate_pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$')
        
    def _preprocess_for_plate_detection(self, crop: np.ndarray) -> np.ndarray:
        """
        Applies morphological preprocessing to enhance plate region visibility.
        Uses bilateral filter (preserves edges while smoothing) + adaptive threshold.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter: smooths noise while keeping plate edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Adaptive threshold to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 9
        )
        
        return thresh

    def _find_plate_contours(self, thresh: np.ndarray, crop_shape: tuple) -> list:
        """
        Finds rectangular contours that are likely license plates.
        
        Filters by:
          - Aspect ratio (1.5 - 5.0 for standard plates)
          - Area (relative to crop size — not too small, not too large)
          - Rectangularity (contour approximation has 4 vertices)
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        crop_h, crop_w = crop_shape[:2]
        crop_area = crop_h * crop_w
        candidates = []
        
        # Sort by area (largest first) for priority
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Plate should be between 1% and 30% of the crop area
            if area < crop_area * 0.01 or area > crop_area * 0.30:
                continue
            
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Plates are rectangular (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / (h + 1e-6)
                
                # Standard plates have aspect ratio between 1.5 and 5.0
                if 1.5 <= aspect_ratio <= 5.5:
                    candidates.append({
                        "contour": approx,
                        "bbox": (x, y, w, h),
                        "area": area,
                        "aspect_ratio": aspect_ratio
                    })
        
        return candidates

    def _perspective_correct(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Applies perspective correction to straighten a skewed plate region.
        Orders the 4 corner points and warps to a flat rectangle.
        """
        pts = contour.reshape(4, 2).astype(np.float32)
        
        # Order points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]     # top-left
        ordered[2] = pts[np.argmax(s)]     # bottom-right
        ordered[1] = pts[np.argmin(d)]     # top-right
        ordered[3] = pts[np.argmax(d)]     # bottom-left
        
        # Calculate output dimensions
        w = int(max(
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[3])
        ))
        h = int(max(
            np.linalg.norm(ordered[3] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1])
        ))
        
        if w < 20 or h < 10:
            return np.array([])
        
        dst = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, M, (w, h))
        
        return warped

    def _ocr_plate(self, plate_image: np.ndarray) -> tuple:
        """
        Runs OCR on a plate image and returns (text, confidence).
        Applies CLAHE enhancement before OCR for better contrast.
        """
        if plate_image.size == 0 or plate_image.shape[0] < 5 or plate_image.shape[1] < 10:
            return "", 0.0
        
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Resize to standard height for consistent OCR
        target_h = 64
        scale = target_h / (gray.shape[0] + 1e-6)
        resized = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        results = self.reader.readtext(resized, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        if results:
            # Combine all detected text segments
            full_text = "".join([r[1] for r in results])
            avg_conf = np.mean([r[2] for r in results])
            
            # Clean up
            clean_text = "".join(e for e in full_text if e.isalnum()).upper()
            return clean_text, avg_conf
        
        return "", 0.0

    def extract_plate_text(self, image: np.ndarray, bbox: list) -> str:
        """
        Full ANPR pipeline:
          1. Crop vehicle region (focus on plate-likely area)
          2. Contour-based plate localization
          3. Perspective correction on best candidate
          4. OCR with CLAHE enhancement
          5. Fallback: direct OCR on heuristic crop if contours fail
        
        Args:
            image: Full frame (BGR)
            bbox: Vehicle bounding box [x1, y1, x2, y2]
        
        Returns:
            Extracted plate string or "UNKNOWN"
        """
        x1, y1, x2, y2 = bbox
        box_h = y2 - y1
        
        # Focus on the bottom portion of the vehicle where plates are located
        y_crop_start = y1 + int(box_h * 0.40)
        y_crop_end = y1 + int(box_h * 0.90)
        
        vehicle_crop = safe_crop(image, x1, y_crop_start, x2, y_crop_end)
        
        if vehicle_crop.size == 0:
            return "UNKNOWN"
        
        best_text = ""
        best_confidence = 0.0
        
        # --- Method 1: Contour-Based Plate Localization ---
        try:
            thresh = self._preprocess_for_plate_detection(vehicle_crop)
            candidates = self._find_plate_contours(thresh, vehicle_crop.shape)
            
            for candidate in candidates[:3]:  # Try top 3 candidates
                # Perspective-correct the plate region
                plate_img = self._perspective_correct(vehicle_crop, candidate["contour"])
                
                if plate_img.size > 0:
                    text, conf = self._ocr_plate(plate_img)
                    if len(text) >= 4 and conf > best_confidence:
                        best_text = text
                        best_confidence = conf
                
                # Also try the raw bounding box crop (sometimes more reliable)
                bx, by, bw, bh = candidate["bbox"]
                raw_plate = safe_crop(vehicle_crop, bx, by, bx + bw, by + bh)
                if raw_plate.size > 0:
                    text, conf = self._ocr_plate(raw_plate)
                    if len(text) >= 4 and conf > best_confidence:
                        best_text = text
                        best_confidence = conf
        except Exception:
            pass  # Fall through to direct OCR
        
        # --- Method 2: Fallback Direct OCR on full crop ---
        if len(best_text) < 4:
            try:
                text, conf = self._ocr_plate(vehicle_crop)
                if len(text) >= 4:
                    best_text = text
                    best_confidence = conf
            except Exception:
                pass
        
        # Validate: plates are usually 6-12 characters
        if 4 <= len(best_text) <= 14:
            return best_text
        
        return "UNKNOWN"
