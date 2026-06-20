"""
GridLock AI - Kaggle Helmet Training Script
Run this script inside a Kaggle Notebook to train your custom helmet detection model using free P100/T4 GPUs.
"""

# Install ultralytics
import os
os.system("pip install ultralytics")

from ultralytics import YOLO
import yaml

# --- 1. Fix the coco128.yaml paths for Kaggle ---
# Automatically find where coco128.yaml is located
yaml_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'coco128.yaml' in files:
        yaml_path = os.path.join(root, 'coco128.yaml')
        break

if not yaml_path:
    raise Exception("Could not find coco128.yaml! Make sure your dataset is added to Kaggle and contains this file.")

dataset_dir = os.path.dirname(yaml_path)

print(f"📁 Found dataset at: {dataset_dir}")

# Read the original YAML
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

# Update paths to be absolute Kaggle paths
config['train'] = os.path.join(dataset_dir, "train/images")
config['val'] = os.path.join(dataset_dir, "val/images")

# Write the fixed YAML to the working directory (since input is read-only on Kaggle)
working_yaml_path = "/kaggle/working/dataset.yaml"
with open(working_yaml_path, 'w') as f:
    yaml.dump(config, f)

print(f"✅ Prepared dataset YAML at: {working_yaml_path}")

# --- 2. Train the Model ---
print("🚀 Starting YOLOv8 training on Kaggle GPU...")

# Load a pre-trained Nano model (fastest, best for edge devices)
model = YOLO('yolov8n.pt') 

# Train the model
results = model.train(
    data=working_yaml_path,
    epochs=50,                  # 50 epochs is a good baseline for hackathons
    imgsz=640,                  # Standard YOLO image size
    batch=32,                   # 32 fits well on Kaggle GPUs
    device=0,                   # Use GPU 0
    name='helmet_model'         # Output directory name
)

print("✅ Training complete!")

# --- 3. Prepare Weights for Download ---
best_weight_path = "/kaggle/working/runs/detect/helmet_model/weights/best.pt"

if os.path.exists(best_weight_path):
    # Copy it to the root working directory so it's easy to download
    os.system(f"cp {best_weight_path} /kaggle/working/helmet_model.pt")
    print("🎉 SUCCESS! Download the 'helmet_model.pt' file from the Kaggle output pane and place it in your GridLock-AI project folder.")
else:
    print("❌ Error: Could not find the trained weights.")
