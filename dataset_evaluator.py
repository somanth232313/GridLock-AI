"""
Dataset Evaluator — GridLock AI
Evaluates the detection engine against ground truth from the Kaggle traffic violation dataset.
Computes proper per-class AP (Average Precision) and mAP@50.
Includes Confusion Matrix and P-R Curve generation data.
"""

import os
import cv2
import json
import numpy as np
from detection_engine import TrafficDetector
from utils import calculate_iou


class DatasetEvaluator:
    def __init__(self, dataset_path="datasetImage"):
        self.dataset_path = dataset_path
        self.val_images_dir = os.path.join(dataset_path, "images", "val")
        self.val_labels_dir = os.path.join(dataset_path, "labels", "val")
        
        # Map Kaggle dataset class IDs to string labels
        self.dataset_class_map = {
            0: 'person',
            1: 'car',
            2: 'truck',
            3: 'bus',
            4: 'motorcycle'
        }
        
        self.detector = TrafficDetector()
        
    def parse_yolo_label(self, label_path: str, img_w: int, img_h: int) -> list:
        """Parses a YOLO format label file and converts normalized coords to absolute bboxes."""
        ground_truth = []
        if not os.path.exists(label_path):
            return ground_truth
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in self.dataset_class_map:
                        label_name = self.dataset_class_map[cls_id]
                        
                        x_center = float(parts[1]) * img_w
                        y_center = float(parts[2]) * img_h
                        width = float(parts[3]) * img_w
                        height = float(parts[4]) * img_h
                        
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        ground_truth.append({
                            "bbox": [x1, y1, x2, y2],
                            "label": label_name
                        })
        return ground_truth

    def _compute_ap(self, precisions: list, recalls: list) -> float:
        """Computes Average Precision using the 11-point interpolation method."""
        if not precisions or not recalls:
            return 0.0
        
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            if prec_at_recall:
                ap += max(prec_at_recall)
        
        ap /= 11.0
        return ap

    def _compute_class_ap(self, all_predictions: list, total_gt: int) -> dict:
        """Computes AP for a single class and returns P-R curve data."""
        if total_gt == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "pr_curve": []}
        
        if not all_predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "pr_curve": []}
        
        # Sort by confidence (highest first)
        all_predictions.sort(key=lambda x: x[0], reverse=True)
        
        precisions = []
        recalls = []
        tp_cumsum = 0
        fp_cumsum = 0
        pr_curve = []
        
        for conf, is_tp in all_predictions:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt
            precisions.append(precision)
            recalls.append(recall)
            pr_curve.append({"recall": recall, "precision": precision, "confidence": conf})
        
        ap = self._compute_ap(precisions, recalls)
        
        final_precision = precisions[-1] if precisions else 0.0
        final_recall = recalls[-1] if recalls else 0.0
        final_f1 = (
            2 * final_precision * final_recall / (final_precision + final_recall)
            if (final_precision + final_recall) > 0 else 0.0
        )
        
        return {
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1,
            "ap": ap,
            "pr_curve": pr_curve
        }

    def evaluate(self, limit: int = 100) -> dict:
        """
        Evaluates the detection engine against ground truth.
        Returns metrics, P-R curves, and Confusion Matrix data.
        """
        if not os.path.exists(self.val_images_dir):
            return {"error": f"Validation images not found at {self.val_images_dir}"}
            
        image_files = [
            f for f in os.listdir(self.val_images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if limit:
            image_files = image_files[:limit]
            
        if not image_files:
            return {"error": "No images found to evaluate."}
        
        class_predictions = {cls: [] for cls in self.dataset_class_map.values()}
        class_gt_counts = {cls: 0 for cls in self.dataset_class_map.values()}
        
        # Initialize confusion matrix (classes + 'background' for FP/FN)
        classes = list(self.dataset_class_map.values())
        cm_size = len(classes) + 1
        confusion_matrix = np.zeros((cm_size, cm_size), dtype=int)
        bg_idx = len(classes)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        for img_name in image_files:
            img_path = os.path.join(self.val_images_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(self.val_labels_dir, label_name)
            
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            h, w = image.shape[:2]
            
            ground_truth = self.parse_yolo_label(label_path, w, h)
            predictions = self.detector.process_image(image, conf_threshold=0.25)
            
            # Count ground truths
            for gt in ground_truth:
                if gt['label'] in class_gt_counts:
                    class_gt_counts[gt['label']] += 1
            
            # Track matched GTs across all predictions in this image to build CM
            matched_gt_indices = set()
            
            # Sort all predictions by confidence
            predictions = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
            
            for pred in predictions:
                pred_label = pred['label']
                if pred_label not in class_to_idx:
                    continue # Ignore classes we don't care about (e.g., traffic lights)
                
                pred_idx = class_to_idx[pred_label]
                best_iou = 0
                best_gt_idx = -1
                best_gt_label = None
                
                for idx, gt in enumerate(ground_truth):
                    if idx in matched_gt_indices:
                        continue
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                        best_gt_label = gt['label']
                
                if best_iou >= 0.5:
                    matched_gt_indices.add(best_gt_idx)
                    gt_idx = class_to_idx.get(best_gt_label, -1)
                    if gt_idx != -1:
                        confusion_matrix[gt_idx][pred_idx] += 1
                        
                        if pred_label == best_gt_label:
                            class_predictions[pred_label].append((pred['confidence'], True))
                        else:
                            # Misclassification (FP for pred_label, FN for best_gt_label handled later)
                            class_predictions[pred_label].append((pred['confidence'], False))
                else:
                    # False positive (predicted something, but no GT matched)
                    confusion_matrix[bg_idx][pred_idx] += 1
                    class_predictions[pred_label].append((pred['confidence'], False))
            
            # Unmatched Ground Truths (False Negatives)
            for idx, gt in enumerate(ground_truth):
                if idx not in matched_gt_indices:
                    gt_label = gt['label']
                    if gt_label in class_to_idx:
                        gt_idx = class_to_idx[gt_label]
                        confusion_matrix[gt_idx][bg_idx] += 1

        # Compute per-class metrics
        metrics = {}
        ap_values = []
        
        for cls_name in classes:
            cls_metrics = self._compute_class_ap(
                class_predictions[cls_name],
                class_gt_counts[cls_name]
            )
            metrics[cls_name] = cls_metrics
            ap_values.append(cls_metrics['ap'])
        
        mean_ap = np.mean(ap_values) if ap_values else 0.0
        
        total_tp = sum(1 for cls in classes for _, is_tp in class_predictions[cls] if is_tp)
        total_fp = sum(1 for cls in classes for _, is_tp in class_predictions[cls] if not is_tp)
        total_fn = sum(class_gt_counts[cls] for cls in classes) - total_tp
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0 else 0.0
        )
        
        # Prepare CM for JSON serialization
        cm_labels = classes + ["background"]
        cm_data = confusion_matrix.tolist()
        
        metrics["overall"] = {
            "mAP_50": mean_ap,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "images_processed": len(image_files)
        }
        
        metrics["confusion_matrix"] = {
            "labels": cm_labels,
            "matrix": cm_data
        }
        
        return metrics


if __name__ == "__main__":
    evaluator = DatasetEvaluator()
    print("Running evaluation on 10 images...")
    results = evaluator.evaluate(limit=10)
    print(json.dumps(results["overall"], indent=2))
