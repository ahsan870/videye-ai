import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile
from pathlib import Path
import time
from collections import defaultdict
import torch
import os
import base64

# Set environment variable to handle MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Set page configuration
st.set_page_config(
    page_title="Car Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated modern styling
st.markdown("""
    <style>
    /* Modern Color Palette */
    :root {
        --primary: #2D3250;
        --secondary: #424769;
        --accent: #676F9D;
        --light: #F6B17A;
        --white: #FFFFFF;
        --gray-light: #F7F7F7;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2rem;
        color: var(--white);
    }
    
    /* Title Styling */
    .title {
        color: var(--white);
        text-align: center;
        padding: 1.5rem 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px var(--shadow);
        margin-bottom: 2rem;
    }
    
    /* Upload Box Styling */
    .upload-container {
        width: 300px;
        margin: 20px 0;
    }
    
    .upload-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 2px dashed var(--light);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .upload-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, var(--light) 0%, #F6B17A 100%);
        color: var(--primary);
        padding: 0.8rem 2rem;
        border-radius: 50px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Info Button Styling */
    .info-button {
        background: var(--light);
        color: var(--primary);
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: none;
        font-size: 1.2rem;
    }
    
    .info-button:hover {
        transform: rotate(180deg);
        background: var(--white);
    }
    
    /* Instructions Panel */
    .instructions {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        color: var(--primary);
        animation: slideDown 0.4s ease-out;
        border-left: 4px solid var(--light);
    }
    
    .instructions h3 {
        color: var(--primary);
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .instructions ol {
        margin-left: 1.5rem;
        line-height: 1.6;
    }
    
    /* Metrics Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card h3 {
        color: var(--light);
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .metric-card h2 {
        color: white;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #F6B17A;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: white;
        margin-top: 0.5rem;
    }
    
    /* DataFrame Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Animation */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background-color: var(--light);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--gray-light);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent);
        border-radius: 4px;
    }
    
    /* Chart Styling */
    .stChart {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Layout Containers */
    .main-content {
        display: flex;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .sidebar {
        flex: 0 0 300px;
    }
    
    .results-area {
        flex: 1;
    }
    
    /* File uploader styling */
    .stFileUploader {
        width: 100%;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 2px dashed var(--light);
        margin-bottom: 1rem;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent);
    }
    
    /* Upload text color */
    .stFileUploader > div {
        color: var(--white) !important;
    }
    
    /* Upload button styling */
    .stFileUploader > button {
        background: var(--light) !important;
        color: var(--primary) !important;
    }
    
    /* Streamlit default elements override */
    .stButton button {
        width: 100%;
        margin-top: 0.5rem;
    }
    
    .css-1x8cf1d {
        width: auto;
    }
    
    .css-10trblm {
        color: var(--white);
        margin-bottom: 1rem;
    }
    
    .logo-container {
        position: absolute;
        top: 20px;
        left: 20px;
        z-index: 1000;
    }
    
    .logo-img {
        width: 150px;
        height: auto;
    }
    
    /* Simple Footer Styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(45, 50, 80, 0.95);
        backdrop-filter: blur(10px);
        padding: 0.8rem 0;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 999;
    }
    
    .footer-text {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Adjust main content to prevent footer overlap */
    .main-content {
        margin-bottom: 60px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize YOLO model
@st.cache_resource
def load_model():
    # Force CPU usage to avoid MPS issues
    model = YOLO('yolov8n.pt')
    model.to('cpu')
    return model

def calculate_movement(previous_center, current_center, threshold=30):
    if previous_center is None:
        return "STRAIGHT"
    
    dx = current_center[0] - previous_center[0]
    if abs(dx) < threshold:
        return "STRAIGHT"
    return "RIGHT" if dx > 0 else "LEFT"

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def track_cars(detections, previous_tracks, frame_id, max_distance=50):
    current_tracks = {}
    used_previous = set()
    
    # For each current detection
    for det in detections:
        bbox = det[:4]
        center = get_center(bbox)
        
        # Find the closest previous track
        min_dist = float('inf')
        best_match = None
        
        for track_id, track_info in previous_tracks.items():
            if track_id in used_previous:
                continue
                
            prev_center = get_center(track_info['bbox'])
            dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                best_match = track_id
        
        if best_match is not None:
            # Update existing track
            current_tracks[best_match] = {
                'bbox': bbox,
                'center': center,
                'last_seen': frame_id,
                'movement': calculate_movement(get_center(previous_tracks[best_match]['bbox']), center)
            }
            used_previous.add(best_match)
        else:
            # Create new track
            new_id = max(list(previous_tracks.keys()) + [0]) + 1
            current_tracks[new_id] = {
                'bbox': bbox,
                'center': center,
                'last_seen': frame_id,
                'movement': "STRAIGHT"
            }
    
    return current_tracks

def process_video(video_path):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    tracks = {}
    car_movements = defaultdict(list)
    frame_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get total frames for progress calculation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Update progress
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        try:
            # Run YOLOv8 detection on CPU
            with torch.no_grad():
                results = model(frame, device='cpu')
            
            # Get detections for cars only (class 2 in COCO dataset)
            detections = []
            for r in results[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if class_id == 2 and score > 0.5:  # Car class with confidence threshold
                    detections.append([x1, y1, x2, y2])
            
            # Track cars
            tracks = track_cars(detections, tracks, frame_count)
            
            # Record movements
            for track_id, track_info in tracks.items():
                car_movements[track_id].append({
                    'frame': frame_count,
                    'movement': track_info['movement'],
                    'timestamp': time.time()
                })
                
        except Exception as e:
            st.error(f"Error processing frame {frame_count}: {str(e)}")
            continue
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    # Process final movements
    final_movements = []
    for car_id, movements in car_movements.items():
        if len(movements) > 5:  # Only consider cars tracked for more than 5 frames
            most_common_movement = max(set(m['movement'] for m in movements), 
                                    key=lambda x: sum(1 for m in movements if m['movement'] == x))
            final_movements.append({
                'car_id': car_id,
                'movement': most_common_movement,
                'timestamp': movements[-1]['timestamp']
            })
    
    return final_movements

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    # Add logo (replace 'logo.png' with your actual logo path)
    logo_path = "logo1.png"  # Update this path
    logo_base64 = img_to_base64(logo_path)
    st.markdown(f"""
        <style>
        .logo-container {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 1000;
        }}
        .logo-img {{
            width: 150px;
            height: auto;
        }}
        </style>
        
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" class="logo-img">
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='title'>Vehicle Movement Analytics</h1>", unsafe_allow_html=True)
    
    # Info button
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = False
    
    # Info button in top right
    _, _, info_col = st.columns([8, 1, 1])
    with info_col:
        if st.button("ℹ️", key="info_button"):
            st.session_state.show_instructions = not st.session_state.show_instructions
    
    if st.session_state.show_instructions:
        st.markdown("""
        <div class="instructions">
        <h3>🎯 Quick Guide</h3>
        <ol>
            <li>📤 Upload your video file (MP4/AVI)</li>
            <li>▶️ Click "Process Video" to start analysis</li>
            <li>⏳ Wait for the processing to complete</li>
            <li>📊 View your detailed analytics report</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    # Main content columns
    upload_col, results_col = st.columns([1, 2])
    
    # Upload section
    with upload_col:
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        
        if uploaded_file is not None:
            if st.button("Process Video"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                
                try:
                    with st.spinner("🔄 Processing..."):
                        movements = process_video(tfile.name)
                        df_movements = pd.DataFrame(movements)
                        st.session_state.results = {
                            'movements': movements,
                            'df': df_movements
                        }
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    Path(tfile.name).unlink()
    
    # Results section
    with results_col:
        if uploaded_file is not None and hasattr(st.session_state, 'results'):
            df_movements = st.session_state.results['df']
            
            st.success("✨ Analysis completed!")
            
            # Calculate movement counts
            total_vehicles = len(df_movements)
            left_turns = len(df_movements[df_movements['movement'] == 'LEFT'])
            right_turns = len(df_movements[df_movements['movement'] == 'RIGHT'])
            straight_moves = len(df_movements[df_movements['movement'] == 'STRAIGHT'])
            
            # Display metrics in a 4-column layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>Total Vehicles</h3>
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Detected</div>
                </div>
                """.format(total_vehicles), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>Left Turns</h3>
                    <div class="metric-value">{}</div>
                    <div class="metric-label">{:.1f}%</div>
                </div>
                """.format(left_turns, (left_turns/total_vehicles)*100 if total_vehicles > 0 else 0), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>Right Turns</h3>
                    <div class="metric-value">{}</div>
                    <div class="metric-label">{:.1f}%</div>
                </div>
                """.format(right_turns, (right_turns/total_vehicles)*100 if total_vehicles > 0 else 0), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>Straight</h3>
                    <div class="metric-value">{}</div>
                    <div class="metric-label">{:.1f}%</div>
                </div>
                """.format(straight_moves, (straight_moves/total_vehicles)*100 if total_vehicles > 0 else 0), unsafe_allow_html=True)
            
            # Movement breakdown chart
            st.markdown("<h2 style='color: var(--white); margin-top: 2rem;'>🔄 Movement Distribution</h2>", unsafe_allow_html=True)
            movement_counts = df_movements['movement'].value_counts()
            st.bar_chart(movement_counts)
            
            # Detailed results
            st.markdown("<h2 style='color: var(--white); margin-top: 2rem;'>📋 Detailed Analysis</h2>", unsafe_allow_html=True)
            st.dataframe(df_movements, use_container_width=True)
            
            csv = df_movements.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis Report",
                data=csv,
                file_name="vehicle_movement_analysis.csv",
                mime="text/csv"
            )

    # Add this at the end of your main function
    st.markdown("""
    <div class="footer">
        <p class="footer-text">© 2024 VidEye. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
