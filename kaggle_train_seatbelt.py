"""
GridLock AI - Kaggle Seatbelt Training Script
Run this script inside a Kaggle Notebook to train your custom seatbelt detection model using free GPUs.

⚠️ IMPORTANT DATASET NOTE ⚠️
The current dataset at `seatbeltdataset` only contains images for a single class ('seat_belt') and lacks YOLO bounding box labels (.txt files). 
To train a YOLOv8 object detection model, your Kaggle dataset must include:
1. Both positive ('seatbelt') and negative ('no_seatbelt') examples (or at least YOLO will learn to detect seatbelts and lack thereof means no seatbelt).
2. YOLO format labels (.txt files) for every image containing bounding box coordinates.
3. A `data.yaml` or `dataset.yaml` file defining the classes.

If you are training an Image Classifier instead of Object Detection, you still need a `no_seatbelt` folder with negative examples.
"""

import os

# Install ultralytics if running on Kaggle
os.system("pip install ultralytics")

from ultralytics import YOLO
import yaml

print("🚀 Starting YOLOv8 Seatbelt Model Training on Kaggle GPU...")

# --- 1. Locate the dataset YAML file ---
yaml_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'data.yaml' in files or 'dataset.yaml' in files:
        yaml_path = os.path.join(root, 'data.yaml' if 'data.yaml' in files else 'dataset.yaml')
        break

if not yaml_path:
    print("❌ Could not find data.yaml or dataset.yaml! Make sure your dataset is uploaded to Kaggle with YOLO labels.")
    # Exit or provide a placeholder for the user to edit
    working_yaml_path = "/kaggle/input/YOUR_DATASET_NAME/data.yaml"
else:
    print(f"📁 Found dataset YAML at: {yaml_path}")
    
    # Read and update paths for Kaggle environment
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_dir = os.path.dirname(yaml_path)
    config['train'] = os.path.join(dataset_dir, "train/images")
    config['val'] = os.path.join(dataset_dir, "val/images")
    
    # Write the fixed YAML to the working directory (since input is read-only on Kaggle)
    working_yaml_path = "/kaggle/working/seatbelt_dataset.yaml"
    with open(working_yaml_path, 'w') as f:
        yaml.dump(config, f)
    print(f"✅ Prepared dataset YAML at: {working_yaml_path}")

# --- 2. Train the Model ---
# Load a pre-trained Nano model (fastest, best for edge devices)
model = YOLO('yolov8n.pt') 

# Train the model (Adjust epochs and batch size as needed)
results = model.train(
    data=working_yaml_path,
    epochs=50,                  # 50 epochs is a good baseline
    imgsz=640,                  # Standard YOLO image size
    batch=32,                   # 32 fits well on Kaggle GPUs
    device=0,                   # Use GPU 0
    name='seatbelt_model'       # Output directory name
)

print("✅ Training complete!")

# --- 3. Prepare Weights for Download ---
best_weight_path = "/kaggle/working/runs/detect/seatbelt_model/weights/best.pt"

if os.path.exists(best_weight_path):
    # Copy it to the root working directory so it's easy to download
    os.system(f"cp {best_weight_path} /kaggle/working/seatbelt_model.pt")
    print("🎉 SUCCESS! Download the 'seatbelt_model.pt' file from the Kaggle output pane and place it in your GridLock-AI project folder.")
else:
    print("❌ Error: Could not find the trained weights.")
