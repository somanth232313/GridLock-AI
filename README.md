# 🚨 Automated Traffic Violation Detection System
**Flipkart Gridlock Hackathon 2.0 Edition**

## 📖 Project Overview
This project is a localized, highly optimized Python prototype designed to detect 7 distinct traffic violations from **single static images**. Instead of relying on complex and compute-heavy video tracking, this system utilizes a blazing-fast Dual-Model Ensemble architecture combined with deterministic spatial heuristics to flag violations, extract license plates, and estimate fine revenue in real-time.

---

## 🛠️ Technology Stack
*   **Computer Vision & Preprocessing:** `OpenCV` (cv2), `numpy`
*   **AI Object Detection:** `ultralytics` (YOLOv8 Dual-Model Ensemble)
*   **Automatic Number Plate Recognition (ANPR):** `easyocr`
*   **Database:** `sqlite3`
*   **Dashboard & UI:** `streamlit`, `pandas`

---

## 🧠 Core Architecture

### 1. Three-Layer Preprocessing Engine (`preprocessing.py`)
Matches the specific flowchart architecture for handling real-world environmental noise:
*   **Layer 1 (Static ROI Masking):** Drops useless pixels outside the target zone.
*   **Layer 2 (Lightness Classifiers):** Dynamically checks luminance. Branches to a Pass-Through/Sharpening filter for bright days, or a Y-Channel LUT Gamma equivalent (CLAHE) for low-light/night conditions.
*   **Layer 3 (Contrast-Preserving Resizer):** Prepares the image for the AI engine while preserving OCR readability.

### 2. Dual-Model Ensemble AI Engine (`detection_engine.py`)
To achieve maximum accuracy without massive compute costs, the system runs two models simultaneously:
*   **Base Model (`yolov8n.pt`):** Generalized COCO dataset knowledge.
*   **Custom Model (`best.pt`):** Fine-tuned on the Kaggle Traffic Violation dataset via Transfer Learning.
*   **Custom NMS Merge:** Pools predictions from both models and uses Intersection over Union (IoU) to eliminate duplicates, keeping the highest confidence score.

### 3. Deterministic Spatial Heuristics (`violation_engine.py`)
Instead of training 7 separate AI models, we use math and spatial relationships on top of the YOLO bounding boxes:
*   **Triple Riding:** Checks if `len(person_boxes)` overlapping a single `motorcycle_box` is >= 3.
*   **Helmet & Seatbelt:** Crops the head/windshield area and uses OpenCV Canny Edge Detection to calculate edge density variance.
*   **Stop-Line & Red-Light:** Checks if the vehicle bounding box crosses dynamic `(x, y)` polygon boundaries defined in `config.json`.
*   **Wrong-Side Driving:** Analyzes vehicle aspect ratios.
*   **Illegal Parking:** Checks intersection with No Parking polygons.

### 4. Business ROI Dashboard (`app.py`)
A production-ready Streamlit interface that proves business value:
*   Live pipeline testing with visual evidence logging.
*   Real-time Estimated Fine Revenue calculation.
*   One-click CSV Export for law enforcement reporting.
*   Built-in Dataset Evaluator to prove mAP/Precision metrics to stakeholders.

---

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the Dashboard:**
    ```bash
    streamlit run app.py
    ```
3.  **Configure Intersections:**
    Open `config.json` to adjust the `(x, y)` coordinates for the Stop Line, Red Light Zone, and No Parking Zone to match your specific camera angles. The app will hot-reload automatically!
