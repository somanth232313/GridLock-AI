import os
import cv2
import numpy as np
from ultralytics import YOLO

from utils import calculate_iou

class TrafficDetector:
    def __init__(self):
        """
        Initializes the Multi-Model Ensemble Engine.
        Loads the generalized COCO model, specialized custom model, 
        and (if available) the dedicated helmet detection model.
        """
        print("Loading Ensemble Models...")
        self.base_model = YOLO("yolov8n.pt")  # Will auto-download if missing
        self.custom_model = YOLO("best.pt")
        
        # Load the custom helmet model trained on Kaggle (if it exists)
        self.helmet_model = None
        if os.path.exists("helmet_model.pt"):
            print("Detected custom helmet model! Loading...")
            self.helmet_model = YOLO("helmet_model.pt")
            self.helmet_classes = [0, 1, 2, 3] # with helmet, without helmet, rider, number plate
            self.helmet_names = self.helmet_model.names
        
        # Mapping COCO indices to string labels
        self.base_classes = [0, 2, 3, 5, 7]
        self.base_names = self.base_model.names
        
        # Mapping Kaggle indices to string labels (added 5: red light, 6: green light)
        self.custom_classes = [0, 1, 2, 3, 4, 5, 6]
        self.custom_names = self.custom_model.names
        
        # Unified target labels for consistency (now including helmet classes)
        self.target_labels = [
            'person', 'car', 'motorcycle', 'bus', 'truck', 
            'red light', 'green light', 
            'with helmet', 'without helmet', 'rider', 'number plate'
        ]

    def _get_detections(self, model, image, conf_threshold, target_classes, names_map):
        results = model.predict(source=image, conf=conf_threshold, classes=target_classes, verbose=False)
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = names_map[cls_id]
                
                if label in self.target_labels:
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "label": label,
                        "confidence": conf
                    })
        return detections

    def process_image(self, image: np.ndarray, conf_threshold: float = 0.4) -> list:
        """
        Runs inference on all active models and merges the results using Non-Maximum Suppression (NMS).
        """
        # 1. Run inference on models
        base_detections = self._get_detections(self.base_model, image, conf_threshold, self.base_classes, self.base_names)
        custom_detections = self._get_detections(self.custom_model, image, conf_threshold, self.custom_classes, self.custom_names)
        
        helmet_detections = []
        if self.helmet_model:
            helmet_detections = self._get_detections(self.helmet_model, image, conf_threshold, self.helmet_classes, self.helmet_names)
        
        # 2. Combine all raw detections into one pool
        all_detections = base_detections + custom_detections + helmet_detections
        
        # 3. Custom Non-Maximum Suppression (NMS)
        # Sort by confidence highest to lowest so we naturally keep the best predictions
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        for det in all_detections:
            keep = True
            for final_det in final_detections:
                # If both models detected the exact same object (high overlap)
                if det['label'] == final_det['label']:
                    iou = calculate_iou(det['bbox'], final_det['bbox'])
                    if iou > 0.45:  # Overlap threshold
                        keep = False
                        break
            if keep:
                final_detections.append(det)
                
        return final_detections
