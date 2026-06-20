import cv2
import numpy as np
import json
import os

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

class ViolationDetector:
    def __init__(self, config_path="config.json"):
        # Load dynamic configuration
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.stop_line_y = config.get("stop_line_y", 500)
                self.red_light_polygon = config.get("red_light_polygon", [(100, 400), (400, 400), (450, 600), (50, 600)])
                self.no_parking_polygon = config.get("no_parking_polygon", [(800, 200), (1000, 200), (1000, 500), (800, 500)])
        else:
            # Fallbacks
            self.stop_line_y = 500  
            self.red_light_polygon = [(100, 400), (400, 400), (450, 600), (50, 600)]
            self.no_parking_polygon = [(800, 200), (1000, 200), (1000, 500), (800, 500)]

    def check_helmet(self, image, person_box):
        """Heuristic: analyzes edge density in the top 25% of the person box."""
        x1, y1, x2, y2 = person_box
        head_y2 = y1 + int((y2 - y1) * 0.25)
        
        # Ensure bounds are safe
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, head_y2 = min(w, x2), min(h, head_y2)
        
        head_crop = image[y1:head_y2, x1:x2]
        
        if head_crop.size == 0:
            return False, 0.0
            
        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density (higher density = hair/face = no helmet)
        edge_density = np.sum(edges > 0) / (head_crop.shape[0] * head_crop.shape[1] + 1e-6)
        
        # Normalize density to a pseudo-confidence score
        no_helmet_confidence = min(1.0, edge_density * 4.0) 
        
        # Threshold for no helmet
        if no_helmet_confidence > 0.6:
            return True, no_helmet_confidence
        return False, 0.0

    def check_seatbelt(self, image, vehicle_box):
        """Heuristic: analyzes horizontal edge variance in windshield area."""
        x1, y1, x2, y2 = vehicle_box
        # Windshield is roughly the top 30-50% of the car box
        w_y1 = y1 + int((y2 - y1) * 0.3)
        w_y2 = y1 + int((y2 - y1) * 0.5)
        
        h, w = image.shape[:2]
        x1, w_y1 = max(0, x1), max(0, w_y1)
        x2, w_y2 = min(w, x2), min(h, w_y2)
        
        windshield = image[w_y1:w_y2, x1:x2]
        
        if windshield.size == 0:
            return False, 0.0
            
        # A seatbelt creates strong diagonal edges. Lack of them implies no seatbelt.
        gray = cv2.cvtColor(windshield, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (windshield.shape[0] * windshield.shape[1] + 1e-6)
        
        no_seatbelt_confidence = 1.0 - min(1.0, edge_density * 3.0)
        if no_seatbelt_confidence > 0.7:
            return True, no_seatbelt_confidence
        return False, 0.0

    def check_wrong_side(self, detection):
        """Heuristic: based on aspect ratio and typical lane assumptions."""
        x1, y1, x2, y2 = detection['bbox']
        aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)
        # If aspect ratio is unusually wide, it might be turning or perpendicular
        if aspect_ratio > 2.0:
            return True, 0.85
        return False, 0.0

    def is_inside_polygon(self, box, polygon):
        x_center = (box[0] + box[2]) / 2
        y_bottom = box[3]
        pts = np.array(polygon, np.int32)
        dist = cv2.pointPolygonTest(pts, (x_center, y_bottom), False)
        return dist >= 0

    def detect_violations(self, detections, image, traffic_light_state="GREEN"):
        violations = []
        people = [d for d in detections if d['label'] == 'person']
        motorcycles = [d for d in detections if d['label'] == 'motorcycle']
        cars_and_trucks = [d for d in detections if d['label'] in ['car', 'bus', 'truck']]
        
        # --- Motorcycle Violations ---
        for bike in motorcycles:
            riders_on_bike = []
            for person in people:
                if calculate_iou(bike['bbox'], person['bbox']) > 0.1:
                    riders_on_bike.append(person)
            
            if len(riders_on_bike) >= 3:
                violations.append({"type": "Triple Riding", "bbox": bike['bbox'], "confidence": 0.95})
            
            for rider in riders_on_bike:
                no_helmet, conf = self.check_helmet(image, rider['bbox'])
                if no_helmet:
                    violations.append({"type": "No Helmet", "bbox": rider['bbox'], "confidence": conf})
        
        # --- Vehicle Violations ---
        for vehicle in cars_and_trucks:
            box = vehicle['bbox']
            no_seatbelt, conf = self.check_seatbelt(image, box)
            if no_seatbelt:
                violations.append({"type": "No Seatbelt", "bbox": box, "confidence": conf})
                
        # --- Spatial & Zone Violations ---
        for vehicle in cars_and_trucks + motorcycles:
            box = vehicle['bbox']
            
            is_wrong_side, conf = self.check_wrong_side(vehicle)
            if is_wrong_side:
                violations.append({"type": "Wrong-Side Driving", "bbox": box, "confidence": conf})
                
            if traffic_light_state == "RED":
                if box[3] > self.stop_line_y and box[1] < self.stop_line_y:
                    violations.append({"type": "Stop-Line Violation", "bbox": box, "confidence": 0.9})
            
            if traffic_light_state == "RED":
                if self.is_inside_polygon(box, self.red_light_polygon):
                    violations.append({"type": "Red-Light Violation", "bbox": box, "confidence": 0.92})
                    
            if self.is_inside_polygon(box, self.no_parking_polygon):
                violations.append({"type": "Illegal Parking", "bbox": box, "confidence": 0.98})

        return violations
