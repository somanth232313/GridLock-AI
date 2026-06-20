"""
Violation Detection Engine — GridLock AI
Uses advanced multi-feature analysis on YOLO bounding boxes to detect 7 violation types.

Helmet Detection uses a 5-feature ensemble:
  1. HSV Color Uniformity
  2. Laplacian Texture Variance
  3. Canny Edge Density
  4. HOG Descriptor Energy (shape structure)
  5. Circular Hough Transform (dome shape detection)
"""

import cv2
import numpy as np
import json
import os

from utils import calculate_iou, box_bottom_center, safe_crop, point_in_polygon


class ViolationDetector:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.load_config()
        
        # Pre-initialize HOG descriptor for helmet detection
        self.hog = cv2.HOGDescriptor(
            _winSize=(32, 32),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
    def load_config(self):
        """Loads spatial configuration from config.json for zone-based detections."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.stop_line_y = config.get("stop_line_y", 500)
                self.red_light_polygon = config.get("red_light_polygon", [[100, 400], [400, 400], [450, 600], [50, 600]])
                self.no_parking_polygon = config.get("no_parking_polygon", [[800, 200], [1000, 200], [1000, 500], [800, 500]])
                
                lane_config = config.get("lanes", {})
                self.left_lane_polygon = lane_config.get("left_lane", [])
                self.right_lane_polygon = lane_config.get("right_lane", [])
                self.expected_direction = lane_config.get("expected_direction", "right")
        else:
            self.stop_line_y = 500
            self.red_light_polygon = [[100, 400], [400, 400], [450, 600], [50, 600]]
            self.no_parking_polygon = [[800, 200], [1000, 200], [1000, 500], [800, 500]]
            self.left_lane_polygon = []
            self.right_lane_polygon = []
            self.expected_direction = "right"

    # ==================================================================================
    #  HELMET DETECTION — 5-Feature Ensemble (HOG + Hough + HSV + Texture + Edge)
    # ==================================================================================
    def check_helmet(self, image: np.ndarray, person_box: list) -> tuple:
        """
        Advanced helmet detection using a 5-feature weighted ensemble:
        
          1. HSV Color Uniformity — helmets are solid-colored; hair/skin is varied
          2. Laplacian Texture Variance — helmets are smooth; hair is textured
          3. Canny Edge Density — helmets produce fewer edges
          4. HOG Descriptor Energy — structured shape vs organic texture
          5. Circular Hough Transform — helmets have dome-like curvature
        
        A weighted vote across all 5 features produces a robust confidence score
        that is significantly more accurate than any single signal.
        
        Returns:
            (is_violation: bool, confidence: float)
        """
        x1, y1, x2, y2 = person_box
        person_h = y2 - y1
        person_w = x2 - x1
        
        # Skip tiny detections where analysis would be unreliable
        if person_h < 40 or person_w < 20:
            return False, 0.0
        
        # Extract head region — top 22% of person bounding box
        head_y2 = y1 + int(person_h * 0.22)
        head_crop = safe_crop(image, x1, y1, x2, head_y2)
        
        if head_crop.size == 0 or head_crop.shape[0] < 8 or head_crop.shape[1] < 8:
            return False, 0.0
        
        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        
        # --- Feature 1: HSV Color Uniformity ---
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        color_uniformity = (h_std + s_std) / 2.0
        # High uniformity std = varied colors = no helmet
        color_score = min(1.0, color_uniformity / 50.0)
        
        # --- Feature 2: Laplacian Texture Variance ---
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # High variance = textured (hair) = no helmet
        texture_score = min(1.0, laplacian_var / 800.0)
        
        # --- Feature 3: Canny Edge Density ---
        edges = cv2.Canny(gray, 50, 150)
        total_pixels = head_crop.shape[0] * head_crop.shape[1]
        edge_density = np.sum(edges > 0) / (total_pixels + 1e-6)
        edge_score = min(1.0, edge_density * 5.0)
        
        # --- Feature 4: HOG Descriptor Energy ---
        # HOG captures gradient orientation patterns. Helmets have structured,
        # directional gradients (smooth curves). Hair has random, high-energy gradients.
        hog_score = 0.5  # neutral default
        try:
            resized_head = cv2.resize(gray, (32, 32))
            hog_features = self.hog.compute(resized_head)
            if hog_features is not None:
                hog_energy = np.mean(np.abs(hog_features))
                # Higher HOG energy = more structural detail = likely no helmet
                # Helmets have low-to-medium energy (smooth dome)
                hog_score = min(1.0, hog_energy * 8.0)
        except Exception:
            pass
        
        # --- Feature 5: Circular Hough Transform (Dome Detection) ---
        # Helmets are dome-shaped. If we detect circular arcs, it suggests a helmet.
        circle_score = 0.7  # Default: assume no helmet (score toward "violation")
        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            min_radius = max(3, min(head_crop.shape[:2]) // 6)
            max_radius = max(min_radius + 1, min(head_crop.shape[:2]) // 2)
            
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=max(5, min(head_crop.shape[:2]) // 3),
                param1=100, param2=25,
                minRadius=min_radius, maxRadius=max_radius
            )
            
            if circles is not None and len(circles[0]) > 0:
                # Circles detected → likely a helmet dome → NOT a violation
                circle_score = 0.15  # Low score = helmet present
            else:
                circle_score = 0.75  # No dome → likely no helmet
        except Exception:
            pass
        
        # --- Weighted Ensemble Vote ---
        # Each score represents "likelihood of NO helmet" (higher = more likely violation)
        weights = [0.20, 0.25, 0.15, 0.20, 0.20]
        combined_score = (
            weights[0] * color_score +
            weights[1] * texture_score +
            weights[2] * edge_score +
            weights[3] * hog_score +
            weights[4] * circle_score
        )
        
        if combined_score > 0.50:
            return True, round(min(0.95, combined_score), 2)
        
        return False, 0.0

    # ==================================================================================
    #  SEATBELT DETECTION — Diagonal Line Analysis + Color Segmentation
    # ==================================================================================
    def check_seatbelt(self, image: np.ndarray, vehicle_box: list) -> tuple:
        """
        Seatbelt detection using diagonal line analysis + dark strap color segmentation.
        
        Method:
          1. Extract windshield/shoulder region of the vehicle
          2. Look for dark diagonal straps using HSV color filtering
          3. Apply Hough Transform to find diagonal lines (25-70 degrees)
          4. Absence of both dark straps AND diagonal lines suggests no seatbelt
        
        Returns:
            (is_violation: bool, confidence: float)
        """
        x1, y1, x2, y2 = vehicle_box
        box_h = y2 - y1
        box_w = x2 - x1
        
        if box_h < 60 or box_w < 60:
            return False, 0.0
        
        # Windshield/shoulder region: top 25-50% of vehicle
        w_y1 = y1 + int(box_h * 0.25)
        w_y2 = y1 + int(box_h * 0.50)
        windshield = safe_crop(image, x1, w_y1, x2, w_y2)
        
        if windshield.size == 0 or windshield.shape[0] < 10 or windshield.shape[1] < 10:
            return False, 0.0
        
        gray = cv2.cvtColor(windshield, cv2.COLOR_BGR2GRAY)
        
        # --- Signal 1: Dark strap detection via HSV ---
        hsv_wind = cv2.cvtColor(windshield, cv2.COLOR_BGR2HSV)
        # Seatbelts are typically dark (black/gray) with low saturation
        dark_mask = cv2.inRange(hsv_wind, (0, 0, 0), (180, 80, 100))
        dark_ratio = np.sum(dark_mask > 0) / (dark_mask.shape[0] * dark_mask.shape[1] + 1e-6)
        
        # --- Signal 2: Diagonal lines via Hough ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, 50, 150)
        
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=30,
            minLineLength=int(min(windshield.shape[:2]) * 0.3),
            maxLineGap=10
        )
        
        diagonal_count = 0
        if lines is not None:
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1 + 1e-6)))
                if 25 <= angle <= 70:
                    diagonal_count += 1
        
        # --- Combined Decision ---
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1] + 1e-6)
        
        if diagonal_count == 0 and dark_ratio < 0.15:
            # No diagonal lines AND no dark straps
            if edge_density > 0.03:  # Windshield has visible content
                return True, 0.74
        elif diagonal_count == 0 and dark_ratio >= 0.15:
            # Dark areas present but no diagonal lines — ambiguous
            if edge_density > 0.05:
                return True, 0.62
        
        return False, 0.0

    # ==================================================================================
    #  WRONG-SIDE DRIVING — Lane Region Analysis
    # ==================================================================================
    def check_wrong_side(self, detection: dict, image_shape: tuple) -> tuple:
        """
        Wrong-side driving detection using configurable lane regions.
        
        Primary: Uses lane polygons from config.json.
        Fallback: Image-center lane divider heuristic.
        
        Returns:
            (is_violation: bool, confidence: float)
        """
        bbox = detection['bbox']
        foot_point = box_bottom_center(bbox)
        img_h, img_w = image_shape[:2]
        
        # Method 1: Configured lane polygons
        if self.left_lane_polygon and self.right_lane_polygon:
            if self.expected_direction == "right":
                if point_in_polygon(foot_point, self.left_lane_polygon):
                    return True, 0.88
            elif self.expected_direction == "left":
                if point_in_polygon(foot_point, self.right_lane_polygon):
                    return True, 0.88
            return False, 0.0
        
        # Method 2: Fallback — center-divider heuristic
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = img_w * img_h
        
        if box_area / image_area < 0.02:
            return False, 0.0
        
        if foot_point[0] < img_w * 0.35:
            return True, 0.65
        
        return False, 0.0

    # ==================================================================================
    #  MAIN DETECTION PIPELINE
    # ==================================================================================
    def detect_violations(self, detections: list, image: np.ndarray, traffic_light_state: str = "GREEN") -> list:
        """
        Main violation detection pipeline. Analyzes YOLO detections to flag 7 types:
          1. No Helmet         — 5-feature ensemble (HOG + Hough + HSV + Texture + Edge)
          2. No Seatbelt       — Diagonal line + dark strap analysis
          3. Triple Riding     — Person count with spatial containment
          4. Wrong-Side Driving — Lane region analysis
          5. Stop-Line Violation — Position vs stop line during red light
          6. Red-Light Violation — Position inside red-light zone during red
          7. Illegal Parking    — Position inside no-parking zone
        """
        violations = []
        
        people = [d for d in detections if d['label'] == 'person']
        motorcycles = [d for d in detections if d['label'] == 'motorcycle']
        cars_and_trucks = [d for d in detections if d['label'] in ['car', 'bus', 'truck']]
        
        # --- Motorcycle Violations ---
        for bike in motorcycles:
            riders_on_bike = []
            for person in people:
                iou = calculate_iou(bike['bbox'], person['bbox'])
                person_foot = box_bottom_center(person['bbox'])
                
                person_in_bike = (
                    bike['bbox'][0] <= person_foot[0] <= bike['bbox'][2] and
                    bike['bbox'][1] <= person_foot[1] <= bike['bbox'][3]
                )
                
                if iou > 0.15 or person_in_bike:
                    riders_on_bike.append(person)
            
            # Triple Riding
            if len(riders_on_bike) >= 3:
                violations.append({
                    "type": "Triple Riding",
                    "bbox": bike['bbox'],
                    "confidence": min(0.95, 0.75 + len(riders_on_bike) * 0.05)
                })
            
            # Helmet check for each rider
            for rider in riders_on_bike:
                no_helmet, conf = self.check_helmet(image, rider['bbox'])
                if no_helmet:
                    violations.append({
                        "type": "No Helmet",
                        "bbox": rider['bbox'],
                        "confidence": conf
                    })
        
        # --- Vehicle Violations ---
        for vehicle in cars_and_trucks:
            box = vehicle['bbox']
            no_seatbelt, conf = self.check_seatbelt(image, box)
            if no_seatbelt:
                violations.append({
                    "type": "No Seatbelt",
                    "bbox": box,
                    "confidence": conf
                })
        
        # --- Spatial & Zone Violations ---
        all_vehicles = cars_and_trucks + motorcycles
        
        for vehicle in all_vehicles:
            box = vehicle['bbox']
            
            # Wrong-Side Driving
            is_wrong_side, conf = self.check_wrong_side(vehicle, image.shape)
            if is_wrong_side:
                violations.append({
                    "type": "Wrong-Side Driving",
                    "bbox": box,
                    "confidence": conf
                })
            
            # Stop-Line Violation
            if traffic_light_state == "RED":
                if box[3] > self.stop_line_y and box[1] < self.stop_line_y:
                    violations.append({
                        "type": "Stop-Line Violation",
                        "bbox": box,
                        "confidence": 0.90
                    })
            
            # Red-Light Violation
            if traffic_light_state == "RED":
                foot = box_bottom_center(box)
                if point_in_polygon(foot, self.red_light_polygon):
                    violations.append({
                        "type": "Red-Light Violation",
                        "bbox": box,
                        "confidence": 0.92
                    })
            
            # Illegal Parking
            foot = box_bottom_center(box)
            if point_in_polygon(foot, self.no_parking_polygon):
                violations.append({
                    "type": "Illegal Parking",
                    "bbox": box,
                    "confidence": 0.95
                })

        return violations
