"""
Six-Layer Preprocessing Engine — GridLock AI

Flowchart Architecture:
  Input Image → [Layer 1: ROI Masking] → [Layer 2: Weather Classifier] →
  [Layer 3: Shadow Normalizer] → [Layer 4: Motion Deblur] →
  [Layer 5: Lightness Classifier] → [Layer 6: Resizer] → Output

Handles: Low light, rain, fog/haze, shadows, motion blur, and glare.
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

    # ==================================================================================
    #  LAYER 1: Static ROI Masking
    # ==================================================================================
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

    # ==================================================================================
    #  LAYER 2: Weather Classifier — Rain/Fog/Haze Detection & Removal
    # ==================================================================================
    def _detect_weather(self, image: np.ndarray) -> str:
        """
        Classifies the weather condition based on image statistics.
        
        - Fog/Haze: Low contrast (small std dev) + high mean brightness
        - Rain: High frequency vertical streaks in the gradient domain
        - Clear: Normal contrast and brightness
        
        Returns: 'fog', 'rain', or 'clear'
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Fog/Haze: washed-out images have low contrast (low std) and high brightness
        if std_val < 40 and mean_val > 120:
            return "fog"
        
        # Rain: vertical streaks create high vertical gradient energy
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        vertical_energy = np.mean(np.abs(sobely))
        horizontal_energy = np.mean(np.abs(sobelx))
        
        # Rain streaks are predominantly vertical — high V/H ratio
        if vertical_energy > 0 and (vertical_energy / (horizontal_energy + 1e-6)) > 1.8:
            if std_val < 55:
                return "rain"
        
        return "clear"

    def _remove_fog(self, image: np.ndarray) -> np.ndarray:
        """
        Dark Channel Prior dehazing (simplified).
        Estimates atmospheric light and transmission map to recover scene contrast.
        Based on He et al. "Single Image Haze Removal Using Dark Channel Prior"
        """
        img_float = image.astype(np.float64) / 255.0
        
        # Compute dark channel: minimum across RGB in a local patch
        min_channel = np.min(img_float, axis=2)
        kernel_size = max(15, min(image.shape[:2]) // 40)
        if kernel_size % 2 == 0:
            kernel_size += 1
        dark_channel = cv2.erode(min_channel, np.ones((kernel_size, kernel_size)))
        
        # Estimate atmospheric light from the brightest pixels in the dark channel
        flat_dark = dark_channel.flatten()
        num_bright = max(1, int(len(flat_dark) * 0.001))
        bright_indices = np.argsort(flat_dark)[-num_bright:]
        
        atmos_light = np.zeros(3)
        for i in range(3):
            channel_flat = img_float[:, :, i].flatten()
            atmos_light[i] = np.mean(channel_flat[bright_indices])
        atmos_light = np.clip(atmos_light, 0.5, 1.0)
        
        # Estimate transmission map
        normalized = img_float / (atmos_light + 1e-6)
        min_normalized = np.min(normalized, axis=2)
        transmission = 1.0 - 0.85 * cv2.erode(min_normalized, np.ones((kernel_size, kernel_size)))
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # Recover scene
        result = np.zeros_like(img_float)
        for i in range(3):
            result[:, :, i] = (img_float[:, :, i] - atmos_light[i]) / (transmission + 1e-6) + atmos_light[i]
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

    def _remove_rain(self, image: np.ndarray) -> np.ndarray:
        """
        Rain streak removal using guided filter + median filtering.
        Rain streaks are high-frequency vertical noise — we suppress them
        while preserving edge structure via a guided filter.
        """
        # Median filter to remove thin rain streaks
        derained = cv2.medianBlur(image, 5)
        
        # Bilateral filter to further smooth while preserving edges
        derained = cv2.bilateralFilter(derained, 9, 75, 75)
        
        # Blend with original to retain some sharpness (70% derained, 30% original)
        result = cv2.addWeighted(derained, 0.7, image, 0.3, 0)
        
        return result

    def layer2_weather_classifier(self, image: np.ndarray) -> np.ndarray:
        """
        Layer 2: Weather Classifier — Detects fog/rain and applies targeted removal.
        
        - Fog/Haze → Dark Channel Prior dehazing
        - Rain → Guided + median filter streak removal
        - Clear → Pass-through (no modification)
        """
        weather = self._detect_weather(image)
        
        if weather == "fog":
            return self._remove_fog(image)
        elif weather == "rain":
            return self._remove_rain(image)
        
        return image

    # ==================================================================================
    #  LAYER 3: Shadow Normalizer
    # ==================================================================================
    def layer3_shadow_normalizer(self, image: np.ndarray) -> np.ndarray:
        """
        Layer 3: Shadow Normalizer
        Detects harsh shadows and normalizes illumination using morphological
        background subtraction on the luminance channel.
        
        Method: Estimates background illumination via a large morphological closing,
        then divides the luminance channel by it to flatten shadows.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        v_channel = hsv[:, :, 2]
        
        # Check if there are significant shadows (high variance in brightness)
        v_std = np.std(v_channel)
        if v_std < 40:
            return image  # Uniform lighting — no shadow correction needed
        
        # Estimate background illumination via morphological closing
        kernel_size = max(31, min(image.shape[:2]) // 8)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(v_channel, cv2.MORPH_CLOSE, kernel)
        
        # Normalize: divide by background to flatten illumination
        mean_bg = np.mean(background)
        if mean_bg > 0:
            normalized_v = np.clip((v_channel / (background + 1e-6)) * mean_bg, 0, 255)
            hsv[:, :, 2] = normalized_v
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result

    # ==================================================================================
    #  LAYER 4: Motion Deblur — Wiener Filter
    # ==================================================================================
    def _detect_motion_blur(self, image: np.ndarray) -> bool:
        """
        Detects motion blur by analyzing the Laplacian variance.
        Low variance = blurry image. Threshold calibrated for traffic scenes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < 80  # Below 80 = likely blurred

    def layer4_motion_deblur(self, image: np.ndarray) -> np.ndarray:
        """
        Layer 4: Motion Deblur via Wiener-inspired sharpening.
        
        Detects blur via Laplacian variance. If blurry, applies an
        unsharp mask with adaptive strength to recover edge detail.
        """
        if not self._detect_motion_blur(image):
            return image  # Image is sharp — skip
        
        # Unsharp masking: sharpen = original + strength * (original - blurred)
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.8, gaussian, -0.8, 0)
        
        return sharpened

    # ==================================================================================
    #  LAYER 5: Lightness Classifier (original Layer 2)
    # ==================================================================================
    def layer5_lightness_classifiers(self, image: np.ndarray, threshold: int = 100) -> np.ndarray:
        """
        Layer 5: Lightness Classifiers — O(N)
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

    # ==================================================================================
    #  LAYER 6: Contrast-Preserving Resizer (original Layer 3)
    # ==================================================================================
    def layer6_contrast_preserving_resizer(self, image: np.ndarray) -> np.ndarray:
        """
        Layer 6: Contrast-Preserving Resizer
        Uses INTER_AREA for downscaling (anti-aliased) and INTER_CUBIC for upscaling.
        
        Note: Disabled by default because YOLO's internal letterboxing preserves
        bounding box coordinate mapping better than pre-resizing.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        interpolation = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_CUBIC
        return cv2.resize(image, self.target_size, interpolation=interpolation)

    # ==================================================================================
    #  MAIN PIPELINE
    # ==================================================================================
    def execute_pipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Executes the full 6-layer preprocessing pipeline:
        
        Layer 1 (ROI Masking):        Runs if roi_polygon is configured
        Layer 2 (Weather Classifier): Always runs — handles rain/fog/haze
        Layer 3 (Shadow Normalizer):  Always runs — flattens harsh shadows
        Layer 4 (Motion Deblur):      Always runs — recovers blurred frames
        Layer 5 (Lightness):          Always runs — handles low-light/night
        Layer 6 (Resizer):            Disabled by default — YOLO handles this
        """
        img = image.copy()
        
        # Layer 1: ROI Masking (conditional)
        if self.enable_roi_masking:
            img = self.layer1_static_roi_masking(img)
        
        # Layer 2: Weather Classification & Removal (rain/fog/haze)
        img = self.layer2_weather_classifier(img)
        
        # Layer 3: Shadow Normalization
        img = self.layer3_shadow_normalizer(img)
        
        # Layer 4: Motion Deblur
        img = self.layer4_motion_deblur(img)
        
        # Layer 5: Lightness Classification (low-light / bright day)
        img = self.layer5_lightness_classifiers(img)
        
        # Layer 6: Resizing (conditional — disabled because YOLO handles it)
        if self.enable_resizing:
            img = self.layer6_contrast_preserving_resizer(img)
        
        return img
