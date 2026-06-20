import cv2
import numpy as np
import easyocr

class PlateReader:
    def __init__(self):
        """
        Initialize EasyOCR reader. 
        Note: set gpu=True if CUDA is available for faster processing.
        """
        self.reader = easyocr.Reader(['en'], gpu=False)
        
    def extract_plate_text(self, image: np.ndarray, bbox: list) -> str:
        """
        Crops the vehicle bounding box, uses morphology to isolate the plate,
        and runs OCR to extract the license plate string.
        """
        x1, y1, x2, y2 = bbox
        
        # Heuristic: License plates are often between 15% to 55% from the bottom of the vehicle.
        # This chops off the road/tires at the extreme bottom, and the windshield/grille at the top.
        box_h = y2 - y1
        y_crop_start = y1 + int(box_h * 0.45) # 55% from bottom
        y_crop_end = y1 + int(box_h * 0.85)   # 15% from bottom
        
        # Ensure bounding box is within image dimensions
        h, w = image.shape[:2]
        x1, y_crop_start = max(0, x1), max(0, y_crop_start)
        x2, y_crop_end = min(w, x2), min(h, y_crop_end)
        
        vehicle_crop = image[y_crop_start:y_crop_end, x1:x2]
        
        if vehicle_crop.size == 0:
            return "UNKNOWN"
            
        # Optional: Morphological operations (e.g. Top-Hat or Black-Hat) 
        # to highlight plate regions before OCR
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # Running EasyOCR on the cropped vehicle
        # In a highly optimized system, we would run a dedicated plate detector 
        # model (e.g. YOLO plate) on this crop first.
        results = self.reader.readtext(gray)
        
        if results:
            # Assume the most confident or largest text is the plate
            text = results[0][1]
            
            # Clean up text (remove spaces, special characters)
            clean_text = "".join(e for e in text if e.isalnum()).upper()
            
            # Basic validation: plates are usually > 4 characters
            if len(clean_text) >= 4:
                return clean_text
            
        return "UNKNOWN"
