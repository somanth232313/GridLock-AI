import streamlit as st
import sqlite3
import pandas as pd
import cv2
import numpy as np
import os
import json

import importlib

# Force Python to reload the modules from disk instead of using memory cache
import preprocessing
import detection_engine
import anpr_engine
importlib.reload(preprocessing)
importlib.reload(detection_engine)
importlib.reload(anpr_engine)

# Import custom modules
from preprocessing import FlowchartPreprocessor
from detection_engine import TrafficDetector
from violation_engine import ViolationDetector
from anpr_engine import PlateReader
from evidence_logger import EvidenceLogger
from dataset_evaluator import DatasetEvaluator

st.set_page_config(page_title="AI Traffic Violation System", layout="wide", page_icon="🚨")

# Initialize models
@st.cache_resource
def load_ensemble_modules():
    return {
        "preprocessor": FlowchartPreprocessor(),
        "detector": TrafficDetector(),
        "violation_engine": ViolationDetector(),
        "anpr": PlateReader(),
        "logger": EvidenceLogger()
    }

modules = load_ensemble_modules()

def load_fines_config():
    if os.path.exists("config.json"):
        with open("config.json", 'r') as f:
            config = json.load(f)
            return config.get("fines", {})
    return {}

def fetch_data():
    if not os.path.exists("violations.db"):
        return pd.DataFrame()
    conn = sqlite3.connect("violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()
    return df

st.title("🚨 Automated Traffic Violation Detection System")
st.markdown("**Flipkart Gridlock Hackathon Edition** | Featuring Deterministic AI Heuristics & ROI Analytics")

# Create Tabs
tab1, tab2 = st.tabs(["🚦 Live Demo Pipeline", "📊 Model Evaluation Benchmarks"])

with tab1:
    # --- SIDEBAR: Upload & Process ---
    st.sidebar.header("Test Pipeline")
    uploaded_file = st.sidebar.file_uploader("Upload Traffic Image", type=["jpg", "png", "jpeg", "webp", "bmp", "tiff"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.sidebar.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
        
        if st.sidebar.button("Run AI Analysis", type="primary"):
            with st.spinner("Processing image through pipeline..."):
                # Ensure config is reloaded to catch any live changes made during the demo
                modules["violation_engine"].load_config()
                
                # 1. Preprocessing (Flowchart Architecture)
                processed_img = modules["preprocessor"].execute_pipeline(image)
                
                # 2. Detection
                detections = modules["detector"].process_image(processed_img)
                
                # 2.5 Dynamic Traffic Light Logic
                # Scan AI detections to see if it actually saw a traffic light in the sky
                current_light_state = "GREEN"  # Default to safe to prevent false tickets
                for d in detections:
                    if d["label"] == "red light":
                        current_light_state = "RED"
                        break  # Red overrides everything
                    elif d["label"] == "green light":
                        current_light_state = "GREEN"
                
                # Show the judges what the AI sees!
                if current_light_state == "RED":
                    st.sidebar.error("🚦 AI DETECTED: RED LIGHT")
                else:
                    st.sidebar.success("🚦 AI DETECTED: GREEN LIGHT (Or None)")
                
                # 3. Violations (Passing dynamic traffic state to heuristics)
                violations = modules["violation_engine"].detect_violations(detections, processed_img, traffic_light_state=current_light_state)
                
                # 4. Process each violation
                for vol in violations:
                    plate_text = modules["anpr"].extract_plate_text(processed_img, vol["bbox"])
                    img_path = modules["logger"].log_violation(
                        processed_img, vol["bbox"], vol["type"], vol["confidence"], plate_text
                    )
                    
                st.sidebar.success(f"Analysis Complete! Detected {len(violations)} violations.")

    # --- MAIN DASHBOARD ---
    st.header("Analytics Dashboard")
    df = fetch_data()

    if not df.empty:
        fines_config = load_fines_config()
        
        # Calculate estimated revenue
        df['estimated_fine'] = df['violation_type'].map(fines_config).fillna(500) # Default 500 if not found
        total_revenue = df['estimated_fine'].sum()

        # Top Level KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Violations Detected", len(df))
        
        known_plates = df[df["plate_number"] != "UNKNOWN"]["plate_number"].nunique()
        col2.metric("Unique Vehicles", known_plates)
        
        avg_conf = df["confidence_score"].mean() * 100
        col3.metric("Avg AI Confidence", f"{avg_conf:.1f}%")
        
        col4.metric("Est. Fine Revenue", f"₹ {total_revenue:,.0f}")
        
        st.markdown("---")
        
        row1_col1, row1_col2 = st.columns([1, 2])
        
        with row1_col1:
            st.subheader("Violation Breakdown")
            violation_counts = df["violation_type"].value_counts()
            st.bar_chart(violation_counts)
            
        with row1_col2:
            st.subheader("Evidence Log")
            
            # Layout for search and export
            search_col, export_col = st.columns([3, 1])
            with search_col:
                search_term = st.text_input("Search by Plate or Violation:")
            
            display_df = df.copy()
            if search_term:
                display_df = display_df[
                    display_df["plate_number"].str.contains(search_term, case=False) |
                    display_df["violation_type"].str.contains(search_term, case=False)
                ]
                
            with export_col:
                st.write("") # Spacing
                st.write("")
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv,
                    file_name='traffic_violations_report.csv',
                    mime='text/csv',
                )

            st.dataframe(display_df[["id", "timestamp", "violation_type", "plate_number", "estimated_fine", "confidence_score"]], use_container_width=True)

        st.markdown("---")
        st.subheader("Recent Evidence Images")
        
        recent_records = df.sort_values(by="id", ascending=False).head(4)
        cols = st.columns(4)
        
        for idx, (_, row) in enumerate(recent_records.iterrows()):
            img_path = row["image_path"]
            if os.path.exists(img_path):
                img_bgr = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                cols[idx].image(img_rgb, caption=f"Plate: {row['plate_number']} | {row['violation_type']}")
    else:
        st.info("No data available yet. Upload a test image in the sidebar to run the pipeline.")

with tab2:
    st.header("Kaggle Dataset Benchmark")
    st.markdown("Evaluate the underlying Object Detection Engine against the Kaggle `traffic-violation-detection-dataset` ground truth to prove baseline accuracy.")
    
    eval_col1, eval_col2 = st.columns([1, 3])
    with eval_col1:
        num_images = st.slider("Number of Images to evaluate", 10, 500, 50, step=10)
        if st.button("Run Evaluation", type="primary", key="eval_btn"):
            st.session_state['eval_running'] = True
            
    with eval_col2:
        if st.session_state.get('eval_running', False):
            with st.spinner(f"Evaluating {num_images} validation images. This takes a moment..."):
                evaluator = DatasetEvaluator()
                results = evaluator.evaluate(limit=num_images)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.success("Evaluation complete!")
                    overall = results["overall"]
                    
                    st.subheader("Overall AI Performance")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("mAP@50 (Approx)", f"{overall['mAP_50']*100:.1f}%")
                    kpi2.metric("Precision", f"{overall['precision']*100:.1f}%")
                    kpi3.metric("Recall", f"{overall['recall']*100:.1f}%")
                    kpi4.metric("Images Processed", overall["images_processed"])
                    
                    st.subheader("Class-Wise Breakdown")
                    class_data = []
                    for cls_name, metrics in results.items():
                        if cls_name != "overall":
                            class_data.append({
                                "Class": cls_name.capitalize(),
                                "Precision": f"{metrics['precision']*100:.1f}%",
                                "Recall": f"{metrics['recall']*100:.1f}%",
                                "F1-Score": f"{metrics['f1']*100:.1f}%"
                            })
                    st.dataframe(pd.DataFrame(class_data), use_container_width=True)
            st.session_state['eval_running'] = False
