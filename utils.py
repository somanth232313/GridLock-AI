"""
Shared utility functions for the GridLock AI Traffic Violation Detection System.
"""

import numpy as np


def calculate_iou(boxA: list, boxB: list) -> float:
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        boxA: [x1, y1, x2, y2] format bounding box.
        boxB: [x1, y1, x2, y2] format bounding box.
    
    Returns:
        IoU score between 0.0 and 1.0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    union = float(boxAArea + boxBArea - interArea)
    iou = interArea / union if union > 0 else 0.0
    return iou


def box_center(bbox: list) -> tuple:
    """Returns the (x_center, y_center) of a bounding box."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def box_bottom_center(bbox: list) -> tuple:
    """Returns the (x_center, y_bottom) of a bounding box — the 'foot' point."""
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


def box_area(bbox: list) -> int:
    """Returns the area of a bounding box in pixels."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def point_in_polygon(point: tuple, polygon: list) -> bool:
    """
    Checks if a point is inside a polygon using OpenCV's pointPolygonTest.
    
    Args:
        point: (x, y) tuple.
        polygon: List of (x, y) vertices.
    
    Returns:
        True if point is inside or on the polygon boundary.
    """
    import cv2
    pts = np.array(polygon, np.int32)
    dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return dist >= 0


def safe_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Safely crops an image region, clamping coordinates to image boundaries.
    
    Returns:
        Cropped image region, or empty array if the crop is invalid.
    """
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    
    return image[y1:y2, x1:x2]
