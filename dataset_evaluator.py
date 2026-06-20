import os
import cv2
import json
import numpy as np
from detection_engine import TrafficDetector

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

class DatasetEvaluator:
    def __init__(self, dataset_path="datasetImage"):
        self.dataset_path = dataset_path
        self.val_images_dir = os.path.join(dataset_path, "images", "val")
        self.val_labels_dir = os.path.join(dataset_path, "labels", "val")
        
        # Map Kaggle dataset classes to our model's string labels
        self.dataset_class_map = {
            0: 'person',
            1: 'car',
            2: 'truck',
            3: 'bus',
            4: 'motorcycle'
        }
        
        self.detector = TrafficDetector()
        
    def parse_yolo_label(self, label_path, img_w, img_h):
        """Parses a YOLO format label file and converts normalized coords to absolute bounding boxes."""
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
                        
                        # YOLO format: class x_center y_center width height (normalized)
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

    def evaluate(self, limit=100):
        """
        Runs evaluation on the dataset and returns metrics.
        Limits to `limit` images for speed during hackathon demo unless None.
        """
        if not os.path.exists(self.val_images_dir):
            return {"error": f"Validation images not found at {self.val_images_dir}"}
            
        image_files = [f for f in os.listdir(self.val_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if limit:
            image_files = image_files[:limit]
            
        if not image_files:
            return {"error": "No images found to evaluate."}
            
        true_positives = {cls: 0 for cls in self.dataset_class_map.values()}
        false_positives = {cls: 0 for cls in self.dataset_class_map.values()}
        false_negatives = {cls: 0 for cls in self.dataset_class_map.values()}
        
        for img_name in image_files:
            img_path = os.path.join(self.val_images_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(self.val_labels_dir, label_name)
            
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            h, w = image.shape[:2]
            
            # Ground truth
            ground_truth = self.parse_yolo_label(label_path, w, h)
            
            # Predictions (Lower conf threshold slightly for evaluation)
            predictions = self.detector.process_image(image, conf_threshold=0.25)
            
            # Match predictions to ground truth
            for cls_name in self.dataset_class_map.values():
                gt_boxes = [gt for gt in ground_truth if gt['label'] == cls_name]
                pred_boxes = [p for p in predictions if p['label'] == cls_name]
                
                matched_gt = set()
                
                # Sort predictions by confidence
                pred_boxes = sorted(pred_boxes, key=lambda x: x.get('confidence', 0), reverse=True)
                
                for pred in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for idx, gt in enumerate(gt_boxes):
                        if idx in matched_gt:
                            continue
                        iou = calculate_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                            
                    if best_iou >= 0.5:
                        true_positives[cls_name] += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        false_positives[cls_name] += 1
                        
                # Unmatched ground truths are false negatives
                false_negatives[cls_name] += len(gt_boxes) - len(matched_gt)

        # Calculate metrics
        metrics = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for cls_name in self.dataset_class_map.values():
            tp = true_positives[cls_name]
            fp = false_positives[cls_name]
            fn = false_negatives[cls_name]
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[cls_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
        # Macro averages
        macro_precision = np.mean([metrics[cls]["precision"] for cls in self.dataset_class_map.values()])
        macro_recall = np.mean([metrics[cls]["recall"] for cls in self.dataset_class_map.values()])
        
        # Micro averages (overall)
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        metrics["overall"] = {
            "mAP_50": macro_precision, # Approx for hackathon
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "images_processed": len(image_files)
        }
        
        return metrics

if __name__ == "__main__":
    evaluator = DatasetEvaluator()
    print("Running evaluation on 10 images...")
    results = evaluator.evaluate(limit=10)
    print(json.dumps(results, indent=2))
