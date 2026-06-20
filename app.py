"""
GridLock AI — Enterprise Traffic Violation Detection System
Flipkart Grid Hackathon — Championship Edition

Features:
  - Single Image / Batch / Video / Webcam Live Stream processing
  - Real-time analytics dashboard with KPIs, Severity Badges, Trend Timeline
  - Repeat Offender Tracking & Risk Scoring
  - Evidence log with search, CSV export, and PDF Report Generation
  - Model evaluation with Confusion Matrix and P-R curves
  - System architecture visualization
"""

import streamlit as st
import sqlite3
import pandas as pd
import cv2
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import base64
import io

# Import custom modules
from preprocessing import FlowchartPreprocessor
from detection_engine import TrafficDetector
from violation_engine import ViolationDetector
from anpr_engine import PlateReader
from evidence_logger import EvidenceLogger, VIOLATION_SEVERITY
from dataset_evaluator import DatasetEvaluator

# Must be the first Streamlit command
st.set_page_config(page_title="GridLock AI", layout="wide", page_icon="🛡️", initial_sidebar_state="expanded")

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 33, 43, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 210, 255, 0.2);
        border: 1px solid rgba(0, 210, 255, 0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8a9bb8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .badge {
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-critical { background: linear-gradient(90deg, #ff0844 0%, #ffb199 100%); }
    .badge-high { background: linear-gradient(90deg, #f83600 0%, #f9d423 100%); }
    .badge-medium { background: linear-gradient(90deg, #f6d365 0%, #fda085 100%); }
    .badge-low { background: linear-gradient(90deg, #84fab0 0%, #8fd3f4 100%); color: #1a1a1a; }
    
    .offender-card {
        background: rgba(42, 26, 26, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 51, 51, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(255, 51, 51, 0.1);
    }
    .offender-name {
        color: #ff6b6b;
        font-size: 1.4rem;
        font-weight: 800;
    }
    .risk-high { color: #ff3333; text-shadow: 0 0 10px rgba(255,51,51,0.5); }
    .risk-medium { color: #ff9933; }
    .risk-low { color: #33cc33; }
    
    /* Sleek Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: rgba(30, 33, 43, 0.4);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
        color: #8a9bb8;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 33, 43, 0.8);
        border-top: 2px solid #00d2ff;
        border-left: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# Initialize models (cached for performance)
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


def load_fines_config() -> dict:
    if os.path.exists("config.json"):
        with open("config.json", 'r') as f:
            config = json.load(f)
            return config.get("fines", {})
    return {}


def fetch_data() -> pd.DataFrame:
    if not os.path.exists("violations.db"):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect("violations.db")
        df = pd.read_sql_query("SELECT * FROM violations", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


def process_frame(image: np.ndarray, conf_thresh: float):
    """Processes a single frame through the full AI pipeline."""
    modules["violation_engine"].load_config()
    
    # 1. Preprocessing
    processed_img = modules["preprocessor"].execute_pipeline(image)
    
    # 2. Object Detection (Multi-Model Ensemble)
    detections = modules["detector"].process_image(processed_img, conf_threshold=conf_thresh)
    
    # 3. Dynamic Traffic Light State
    current_light_state = "GREEN"
    for d in detections:
        if d["label"] == "red light":
            current_light_state = "RED"
            break
        elif d["label"] == "green light":
            current_light_state = "GREEN"
    
    # 4. Violation Detection (ML + 5-Feature Ensemble)
    violations = modules["violation_engine"].detect_violations(
        detections, processed_img, traffic_light_state=current_light_state
    )
    
    # 5. ANPR + Evidence Logging
    annotated_img_path = None
    for vol in violations:
        plate_text = modules["anpr"].extract_plate_text(processed_img, vol["bbox"])
        annotated_img_path = modules["logger"].log_violation(
            processed_img, vol["bbox"], vol["type"],
            vol["confidence"], plate_text
        )
    
    if not violations or not annotated_img_path:
        return processed_img, violations, detections
    
    return cv2.imread(annotated_img_path), violations, detections


def generate_pdf_report(df: pd.DataFrame, fines_config: dict) -> bytes:
    """Generates a text-based PDF-style report as downloadable content."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("       GRIDLOCK AI - TRAFFIC VIOLATION REPORT")
    report_lines.append(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Summary Statistics
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"  Total Violations Detected:  {len(df)}")
    
    known = df[df["plate_number"] != "UNKNOWN"]["plate_number"].nunique()
    report_lines.append(f"  Unique Vehicles Identified: {known}")
    
    avg_conf = df["confidence_score"].mean() * 100
    report_lines.append(f"  Average AI Confidence:      {avg_conf:.1f}%")
    
    df['est_fine'] = df['violation_type'].map(fines_config).fillna(500)
    total_rev = df['est_fine'].sum()
    report_lines.append(f"  Estimated Fine Revenue:     Rs. {total_rev:,.0f}")
    report_lines.append("")
    
    # Violation Breakdown
    report_lines.append("VIOLATION BREAKDOWN")
    report_lines.append("-" * 40)
    for vtype, count in df["violation_type"].value_counts().items():
        sev = VIOLATION_SEVERITY.get(vtype, {}).get("level", "MEDIUM")
        fine = fines_config.get(vtype, 500)
        report_lines.append(f"  {vtype:<25} Count: {count:>3}  |  Severity: {sev:<10}  |  Fine: Rs. {fine}")
    report_lines.append("")
    
    # Repeat Offenders
    plate_counts = df[df["plate_number"] != "UNKNOWN"].groupby("plate_number").size().sort_values(ascending=False)
    repeat_offenders = plate_counts[plate_counts >= 2]
    
    if len(repeat_offenders) > 0:
        report_lines.append("REPEAT OFFENDERS (2+ Violations)")
        report_lines.append("-" * 40)
        for plate, count in repeat_offenders.items():
            total_fine = df[df["plate_number"] == plate]["est_fine"].sum()
            report_lines.append(f"  Plate: {plate:<15}  Violations: {count:>2}  |  Total Fines: Rs. {total_fine:,.0f}")
        report_lines.append("")
    
    # Individual Records
    report_lines.append("DETAILED EVIDENCE LOG")
    report_lines.append("-" * 40)
    for _, row in df.iterrows():
        sev = row.get("severity", "MEDIUM")
        report_lines.append(f"  [{row['timestamp']}] {row['violation_type']:<22} | Plate: {row['plate_number']:<12} | Conf: {row['confidence_score']:.0%} | Severity: {sev}")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("  Report generated by GridLock AI - Flipkart Grid Hackathon 2.0")
    report_lines.append("=" * 70)
    
    return "\n".join(report_lines).encode("utf-8")


# ==============================================================================================
# MAIN UI
# ==============================================================================================

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00d2ff;'>🛡️ GridLock AI</h2>", unsafe_allow_html=True)
    st.markdown("### :material/settings: Engine Controls")
    
    st.markdown("---")
    input_mode = st.radio(":material/sensors: Input Source", ["Single Image", "Batch Processing", "Video File", "Webcam Live"])
    st.markdown("---")
    conf_threshold = st.slider(":material/tune: AI Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    
    st.markdown("---")
    st.markdown("### :material/memory: AI Subsystems")
    st.success(":material/check_circle: Multi-Model Ensemble")
    st.success(":material/check_circle: Contour ANPR Engine")
    if os.path.exists("helmet_model.pt"):
        st.success(":material/check_circle: Custom Helmet ML")
    else:
        st.warning(":material/pending: 5-Feature Helmet Heuristic")

st.title(":material/warning: GridLock AI")
st.markdown("**Enterprise Traffic Violation Detection** | Multi-Model Ensemble | Contour ANPR | Live Intelligence")
st.markdown("---")

# Create Tabs
tab_pipeline, tab_dash, tab_offenders, tab_eval, tab_arch = st.tabs([
    ":material/traffic: Live Pipeline", ":material/bar_chart: Intelligence Dashboard", ":material/search: Repeat Offenders", ":material/science: Evaluation Matrix", ":material/settings: Architecture"
])

# ==============================================================================================
# TAB 1: LIVE PIPELINE
# ==============================================================================================
with tab_pipeline:
    st.markdown("### Real-Time Detection Feed")
    
    # Now that inputs are in the sidebar, we use the full width for the output/upload area
    
    if input_mode == "Single Image":
        uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg", "png", "jpeg", "webp"])
        if uploaded_file and st.button("Run Analysis", type="primary"):
            with st.spinner("Processing through Multi-Model Ensemble..."):
                try:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    if image is not None:
                        ann_img, violations, detections = process_frame(image, conf_threshold)
                        
                        st.image(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB),
                                 caption=f"{len(detections)} Objects Detected | {len(violations)} Violations Found",
                                 use_container_width=True)
                        
                        if violations:
                            st.markdown("### :material/local_police: Violations Detected")
                            for v in violations:
                                sev = VIOLATION_SEVERITY.get(v['type'], {}).get('level', 'MEDIUM')
                                if sev == "CRITICAL":
                                    st.error(f":material/warning: **{v['type']}** — Severity: CRITICAL (Conf: {v['confidence']:.0%})")
                                elif sev == "HIGH":
                                    st.warning(f":material/warning: **{v['type']}** — Severity: HIGH (Conf: {v['confidence']:.0%})")
                                else:
                                    st.info(f":material/info: **{v['type']}** — Severity: {sev} (Conf: {v['confidence']:.0%})")
                        else:
                            st.success(":material/check_circle: No violations detected in this image.")
                    else:
                        st.error("Invalid image file.")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                        
    elif input_mode == "Batch Processing":
        uploaded_files = st.file_uploader("Upload Multiple Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        if uploaded_files and st.button("Run Batch Analysis", type="primary"):
            total_violations = 0
            total_detections = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing image {i+1}/{len(uploaded_files)}...")
                try:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    if image is not None:
                        _, viols, dets = process_frame(image, conf_threshold)
                        total_violations += len(viols)
                        total_detections += len(dets)
                except Exception:
                    pass
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success(f"Batch complete!")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Images Processed", len(uploaded_files))
            bc2.metric("Objects Detected", total_detections)
            bc3.metric("Violations Found", total_violations)
                    
    elif input_mode == "Video File":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if video_file and st.button("Process Video", type="primary"):
            with st.spinner("Processing video frames..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_file.read())
                tfile.flush()
                vf = cv2.VideoCapture(tfile.name)
                
                stframe = st.empty()
                status = st.empty()
                total_v = 0
                frame_skip = 5
                frame_count = 0
                
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue
                    try:
                        ann_img, viols, _ = process_frame(frame, conf_threshold)
                        total_v += len(viols)
                        stframe.image(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        status.text(f"Frame {frame_count} | Violations so far: {total_v}")
                    except Exception:
                        continue
                
                vf.release()
                os.unlink(tfile.name)
                st.success(f"Video complete! Processed {frame_count} frames, logged {total_v} violations.")
                
    elif input_mode == "Webcam Live":
        st.info(":material/videocam: Click 'Start Webcam' to begin real-time analysis. Press 'Stop' to end.")
        
        if st.button("Start Webcam", type="primary"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            stop_btn = st.button("Stop Webcam")
            
            if cap.isOpened():
                frame_count = 0
                total_v = 0
                status = st.empty()
                
                while cap.isOpened() and not stop_btn:
                    ret, cap_frame = cap.read()
                    if not ret:
                        st.warning("Could not read from webcam.")
                        break
                    
                    frame_count += 1
                    # Process every 10th frame for real-time performance
                    if frame_count % 10 == 0:
                        try:
                            ann_img, viols, _ = process_frame(cap_frame, conf_threshold)
                            total_v += len(viols)
                            stframe.image(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                            status.text(f"Live | Frame {frame_count} | Violations: {total_v}")
                        except Exception:
                            stframe.image(cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    else:
                        stframe.image(cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                cap.release()
            else:
                st.error("Could not access webcam. Make sure a camera is connected.")

# ==============================================================================================
# TAB 2: ANALYTICS DASHBOARD
# ==============================================================================================
with tab_dash:
    df = fetch_data()

    if not df.empty:
        # Compatibility with older database schema
        if "severity" not in df.columns:
            df["severity"] = "MEDIUM"
            
        fines_config = load_fines_config()
        df['estimated_fine'] = df['violation_type'].map(fines_config).fillna(500)
        total_revenue = df['estimated_fine'].sum()

        # Premium Metric Cards — Row 1
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Violations</div>
        </div>
        """, unsafe_allow_html=True)
        
        known_plates = df[df["plate_number"] != "UNKNOWN"]["plate_number"].nunique()
        col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{known_plates}</div>
            <div class="metric-label">Identified Vehicles</div>
        </div>
        """, unsafe_allow_html=True)
        
        avg_conf = df["confidence_score"].mean() * 100
        col3.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_conf:.1f}%</div>
            <div class="metric-label">Avg AI Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        repeat_count = df[df["plate_number"] != "UNKNOWN"].groupby("plate_number").size()
        repeat_offenders_count = len(repeat_count[repeat_count >= 2])
        col4.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ff6b6b;">{repeat_offenders_count}</div>
            <div class="metric-label">Repeat Offenders</div>
        </div>
        """, unsafe_allow_html=True)
        
        col5.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #00ff88;">Rs {total_revenue:,.0f}</div>
            <div class="metric-label">Est. Revenue</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Charts Row
        chart_col1, chart_col2, chart_col3 = st.columns([1, 1, 1])
        
        with chart_col1:
            st.subheader("Violation Distribution")
            fig = px.pie(df, names='violation_type', hole=0.4, template="plotly_dark",
                         color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            st.subheader("Severity Breakdown")
            sev_counts = df["severity"].value_counts()
            sev_colors = {"CRITICAL": "#b30000", "HIGH": "#ff3333", "MEDIUM": "#ff9933", "LOW": "#33b5e5"}
            fig_sev = px.bar(
                x=sev_counts.index, y=sev_counts.values,
                color=sev_counts.index,
                color_discrete_map=sev_colors,
                template="plotly_dark",
                labels={"x": "Severity", "y": "Count"}
            )
            fig_sev.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig_sev, use_container_width=True)
        
        with chart_col3:
            st.subheader("Detection Timeline")
            if "timestamp" in df.columns:
                df_timeline = df.copy()
                df_timeline["timestamp"] = pd.to_datetime(df_timeline["timestamp"], errors="coerce")
                df_timeline = df_timeline.dropna(subset=["timestamp"])
                if not df_timeline.empty:
                    df_timeline["hour"] = df_timeline["timestamp"].dt.strftime("%H:00")
                    hourly = df_timeline.groupby("hour").size().reset_index(name="count")
                    fig_time = px.area(hourly, x="hour", y="count", template="plotly_dark",
                                       color_discrete_sequence=["#00d2ff"])
                    fig_time.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300,
                                          xaxis_title="Hour", yaxis_title="Violations")
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("Timeline data unavailable.")
        
        st.markdown("---")
        
        # Evidence Registry
        st.subheader("Evidence Registry")
        
        search_col, export_col, pdf_col = st.columns([3, 1, 1])
        with search_col:
            search_term = st.text_input(":material/search: Search by Plate, Violation, or Severity:")
        
        display_df = df.copy()
        if search_term:
            display_df = display_df[
                display_df["plate_number"].str.contains(search_term, case=False, na=False) |
                display_df["violation_type"].str.contains(search_term, case=False, na=False) |
                display_df["severity"].str.contains(search_term, case=False, na=False)
            ]
            
        with export_col:
            st.write("")
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=":material/download: CSV Export",
                data=csv,
                file_name='traffic_violations_report.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with pdf_col:
            st.write("")
            pdf_data = generate_pdf_report(df, fines_config)
            st.download_button(
                label=":material/description: Full Report",
                data=pdf_data,
                file_name=f'GridLock_Report_{datetime.now().strftime("%Y%m%d")}.txt',
                mime='text/plain',
                use_container_width=True
            )

        # Styled dataframe
        def style_severity(val):
            colors = {'CRITICAL': 'color: #ff4d4d; font-weight: bold;', 
                      'HIGH': 'color: #ff9933; font-weight: bold;',
                      'MEDIUM': 'color: #ffcc00;',
                      'LOW': 'color: #33cc33;'}
            return colors.get(val, '')

        show_cols = ["timestamp", "violation_type", "severity", "plate_number", "estimated_fine", "confidence_score"]
        available_cols = [c for c in show_cols if c in display_df.columns]
        styled_df = display_df[available_cols].style.map(style_severity, subset=['severity'] if 'severity' in available_cols else [])
        st.dataframe(styled_df, use_container_width=True, height=300)

        st.markdown("---")
        st.subheader(":material/photo_library: Latest Evidence")
        
        recent_records = df.sort_values(by="id", ascending=False).head(4)
        cols = st.columns(4)
        
        for idx, (_, row) in enumerate(recent_records.iterrows()):
            img_path = row["image_path"]
            if os.path.exists(img_path):
                try:
                    img_bgr = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    sev = row.get('severity', 'MEDIUM')
                    sev_class = f"badge-{sev.lower()}"
                    caption_html = f"""
                    <div style='padding: 10px; background: #1e2127; border-radius: 5px;'>
                        <span class='badge {sev_class}'>{sev}</span><br/>
                        <span style='color: #00d2ff; font-weight: bold;'>{row['plate_number']}</span><br/>
                        <span style='color: #a0aabf; font-size: 0.85em;'>{row['violation_type']}</span>
                    </div>
                    """
                    cols[idx].image(img_rgb, use_container_width=True)
                    cols[idx].markdown(caption_html, unsafe_allow_html=True)
                except Exception:
                    cols[idx].warning("Image load failed.")
    else:
        st.info("No data yet. Use the Live Pipeline to process images or video.")

# ==============================================================================================
# TAB 3: REPEAT OFFENDER TRACKING
# ==============================================================================================
with tab_offenders:
    st.header(":material/search: Repeat Offender Intelligence")
    st.markdown("Identifies vehicles with **2 or more violations** and calculates a risk score.")
    
    df_off = fetch_data()
    
    if not df_off.empty:
        if "severity" not in df_off.columns:
            df_off["severity"] = "MEDIUM"
        
        fines_config = load_fines_config()
        df_off['estimated_fine'] = df_off['violation_type'].map(fines_config).fillna(500)
        
        # Filter to known plates with 2+ violations
        known_plates_df = df_off[df_off["plate_number"] != "UNKNOWN"]
        plate_groups = known_plates_df.groupby("plate_number")
        
        offender_data = []
        for plate, group in plate_groups:
            count = len(group)
            if count >= 2:
                total_fine = group["estimated_fine"].sum()
                # Risk Score: based on count + severity
                severity_weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
                sev_score = sum(severity_weights.get(s, 2) for s in group["severity"])
                risk_score = min(100, int((count * 15) + (sev_score * 3)))
                
                violations_list = group["violation_type"].value_counts().to_dict()
                last_seen = group["timestamp"].max()
                
                offender_data.append({
                    "Plate": plate,
                    "Violations": count,
                    "Risk Score": risk_score,
                    "Total Fines": f"Rs {total_fine:,.0f}",
                    "Last Seen": last_seen,
                    "Types": ", ".join([f"{v}({c})" for v, c in violations_list.items()])
                })
        
        if offender_data:
            # Sort by risk score
            offender_data = sorted(offender_data, key=lambda x: x["Risk Score"], reverse=True)
            
            # Summary KPIs
            ko1, ko2, ko3 = st.columns(3)
            ko1.metric("Repeat Offenders", len(offender_data))
            total_repeat_fines = sum(float(o["Total Fines"].replace("Rs ", "").replace(",", "")) for o in offender_data)
            ko2.metric("Total Fines from Repeats", f"Rs {total_repeat_fines:,.0f}")
            avg_risk = np.mean([o["Risk Score"] for o in offender_data])
            ko3.metric("Avg Risk Score", f"{avg_risk:.0f}/100")
            
            st.markdown("---")
            
            # Display offender cards
            for offender in offender_data:
                risk = offender["Risk Score"]
                if risk >= 70:
                    risk_class = "risk-high"
                    risk_label = "HIGH RISK"
                elif risk >= 40:
                    risk_class = "risk-medium"
                    risk_label = "MEDIUM RISK"
                else:
                    risk_class = "risk-low"
                    risk_label = "LOW RISK"
                
                st.markdown(f"""
                <div class="offender-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="offender-name">:material/directions_car: {offender['Plate']}</span>
                            <span class="badge badge-high" style="margin-left: 10px;">{offender['Violations']} violations</span>
                        </div>
                        <div>
                            <span class="{risk_class}" style="font-size: 1.5rem; font-weight: 700;">{risk}/100</span>
                            <br/><span style="color: #a0aabf; font-size: 0.8rem;">{risk_label}</span>
                        </div>
                    </div>
                    <div style="margin-top: 10px; color: #a0aabf;">
                        <strong>Total Fines:</strong> {offender['Total Fines']} | 
                        <strong>Last Seen:</strong> {offender['Last Seen']} | 
                        <strong>Types:</strong> {offender['Types']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Offender data table
            st.markdown("---")
            st.subheader("Offender Data Table")
            st.dataframe(pd.DataFrame(offender_data), use_container_width=True)
        else:
            st.info("No repeat offenders found yet. Vehicles need 2+ violations to appear here.")
    else:
        st.info("No data yet. Process images to build the offender database.")

# ==============================================================================================
# TAB 4: MODEL EVALUATION
# ==============================================================================================
with tab_eval:
    st.header("AI Model Evaluation Benchmarks")
    st.markdown("PASCAL VOC 11-point AP | Confusion Matrix | Precision-Recall Curves")
    
    eval_col1, eval_col2 = st.columns([1, 4])
    with eval_col1:
        num_images = st.slider("Validation Images", 10, 500, 50, step=10)
        if st.button("Run Evaluation", type="primary", use_container_width=True):
            st.session_state['eval_running'] = True
            
    with eval_col2:
        if st.session_state.get('eval_running', False):
            try:
                with st.spinner(f"Evaluating {num_images} images..."):
                    evaluator = DatasetEvaluator()
                    results = evaluator.evaluate(limit=num_images)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        overall = results["overall"]
                        
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("mAP@50", f"{overall['mAP_50']*100:.1f}%")
                        m2.metric("Precision", f"{overall['precision']*100:.1f}%")
                        m3.metric("Recall", f"{overall['recall']*100:.1f}%")
                        m4.metric("F1-Score", f"{overall['f1']*100:.1f}%")
                        m5.metric("Images", overall["images_processed"])
                        
                        st.markdown("---")
                        
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            st.subheader("Confusion Matrix")
                            cm_data = results["confusion_matrix"]
                            fig_cm = px.imshow(
                                cm_data["matrix"],
                                x=cm_data["labels"],
                                y=cm_data["labels"],
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                text_auto=True,
                                color_continuous_scale="Blues",
                                template="plotly_dark"
                            )
                            fig_cm.update_layout(height=400)
                            st.plotly_chart(fig_cm, use_container_width=True)
                            
                        with chart_col2:
                            st.subheader("Precision-Recall Curves")
                            fig_pr = go.Figure()
                            
                            for cls_name, metrics in results.items():
                                if cls_name not in ["overall", "confusion_matrix"] and "pr_curve" in metrics:
                                    pr_data = metrics["pr_curve"]
                                    if pr_data:
                                        recalls = [pt["recall"] for pt in pr_data]
                                        precisions = [pt["precision"] for pt in pr_data]
                                        points = sorted(zip(recalls, precisions))
                                        if points:
                                            r, p = zip(*points)
                                            fig_pr.add_trace(go.Scatter(
                                                x=list(r), y=list(p), mode='lines',
                                                name=f"{cls_name} (AP: {metrics['ap']:.2f})"
                                            ))
                            
                            fig_pr.update_layout(
                                xaxis_title="Recall", yaxis_title="Precision",
                                xaxis=dict(range=[0, 1.05]),
                                yaxis=dict(range=[0, 1.05]),
                                template="plotly_dark",
                                hovermode="x unified",
                                height=400
                            )
                            st.plotly_chart(fig_pr, use_container_width=True)
                            
                        st.subheader("Per-Class AP")
                        class_data = []
                        for cls_name, cls_metrics in results.items():
                            if cls_name not in ["overall", "confusion_matrix"]:
                                class_data.append({
                                    "Class": cls_name.capitalize(),
                                    "AP@50": f"{cls_metrics.get('ap', 0)*100:.1f}%",
                                    "Precision": f"{cls_metrics.get('precision', 0)*100:.1f}%",
                                    "Recall": f"{cls_metrics.get('recall', 0)*100:.1f}%",
                                    "F1": f"{cls_metrics.get('f1', 0)*100:.1f}%"
                                })
                        st.dataframe(pd.DataFrame(class_data), use_container_width=True)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
            finally:
                st.session_state['eval_running'] = False
        else:
            st.info("Click 'Run Evaluation' to compute mAP, Confusion Matrix, and PR curves.")

# ==============================================================================================
# TAB 5: SYSTEM ARCHITECTURE
# ==============================================================================================
with tab_arch:
    st.header("System Architecture")
    
    col_diagram, col_details = st.columns([2, 1])
    
    with col_diagram:
        st.markdown("""
        ### End-to-End Pipeline
        The system processes input through a multi-stage pipeline optimized for accuracy and speed.
        """)
        
        st.image("architecture.png", use_container_width=True, caption="GridLock AI System Architecture")
        
    with col_details:
        st.markdown("""
        ### Key Differentiators
        
        **1. 5-Feature Helmet Ensemble**
        HOG descriptor energy + Circular Hough dome detection + HSV color uniformity + Laplacian texture + Canny edges. Weighted vote across all 5 signals.
        
        **2. Dual-Model Ensemble**
        COCO-pretrained `yolov8n.pt` + Kaggle-trained `best.pt` merged via custom IoU-NMS.
        
        **3. Contour-Based ANPR**
        Bilateral filter → adaptive threshold → contour polygon approximation → perspective warp → CLAHE → EasyOCR.
        
        **4. Repeat Offender Intelligence**
        Tracks vehicles across sessions. Calculates risk scores based on violation count and severity.
        
        **5. PASCAL VOC Evaluation**
        11-point interpolation AP, Confusion Matrix, and interactive P-R curves.
        
        **6. Multi-Input Support**
        Single image, batch, video file, and live webcam processing.
        """)
