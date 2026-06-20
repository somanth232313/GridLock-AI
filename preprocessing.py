"""
Three-Layer Preprocessing Engine — GridLock AI

Flowchart Architecture:
  Input Image → [Layer 1: ROI Masking] → [Layer 2: Lightness Classifier] → [Layer 3: Resizer] → Output

Each layer can be enabled/disabled based on the deployment context.
"""

import cv2
import numpy as np


class FlowchartPreprocessor:
    def __init__(self):
        # Layer configuration — can be adjusted per-deployment
        self.enable_roi_masking = False  # Enable when ROI polygon is configured
        self.enable_resizing = False     # Disabled: YOLO handles dynamic resizing internally
        self.roi_polygon = None          # Set via configure_roi()
        self.target_size = (640, 640)    # YOLO default input size

    def configure_roi(self, polygon_points: list):
        """
        Configure and enable ROI masking with the given polygon.
        Call this when you know the camera's region of interest.
        """
        if polygon_points and len(polygon_points) >= 3:
            self.roi_polygon = polygon_points
            self.enable_roi_masking = True

    def layer1_static_roi_masking(self, image: np.ndarray) -> np.ndarray:
        """
        Layer 1: Static ROI Masking — O(1) per-pixel
        Drops ~40% of useless pixels outside the defined region of interest.
        This reduces noise from sky, buildings, and irrelevant background.
        """
        if self.roi_polygon is None:
            return image
            
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array(self.roi_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    def layer2_lightness_classifiers(self, image: np.ndarray, threshold: int = 100) -> np.ndarray:
        """
        Layer 2: Lightness Classifiers — O(N)
        Checks ambient illumination via Y-channel (luminance) and branches:
          - Bright Day Path → Sharpening kernel (mitigates motion blur)
          - Low-Light Path  → CLAHE on Y-channel (enhances shadows/night)
        """
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        
        avg_luminance = np.mean(y_channel)
        
        if avg_luminance >= threshold:
            return self._bright_day_path(image)
        else:
            return self._low_light_path(image, yuv, y_channel)

    def _bright_day_path(self, image: np.ndarray) -> np.ndarray:
        """Applies sharpness kernel to mitigate motion blur on bright days."""
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def _low_light_path(self, image: np.ndarray, yuv: np.ndarray, y_channel: np.ndarray) -> np.ndarray:
        """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on Y-channel."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(y_channel)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def layer3_contrast_preserving_resizer(self, image: np.ndarray) -> np.ndarray:
        """
        Layer 3: Contrast-Preserving Resizer
        Uses INTER_AREA for downscaling (anti-aliased) and INTER_CUBIC for upscaling.
        
        Note: Disabled by default because YOLO's internal letterboxing preserves
        bounding box coordinate mapping better than pre-resizing.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        interpolation = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_CUBIC
        return cv2.resize(image, self.target_size, interpolation=interpolation)

    def execute_pipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Executes the full preprocessing pipeline based on enabled layers.
        
        Layer 1 (ROI Masking):  Runs if roi_polygon is configured via configure_roi()
        Layer 2 (Lightness):    Always runs — essential for handling varied lighting
        Layer 3 (Resizer):      Disabled by default — YOLO handles this internally
        """
        img = image.copy()
        
        # Layer 1: ROI Masking (conditional)
        if self.enable_roi_masking:
            img = self.layer1_static_roi_masking(img)
        
        # Layer 2: Lightness Classification (always active)
        img = self.layer2_lightness_classifiers(img)
        
        # Layer 3: Resizing (conditional — disabled because YOLO handles it)
        if self.enable_resizing:
            img = self.layer3_contrast_preserving_resizer(img)
        
        return img
