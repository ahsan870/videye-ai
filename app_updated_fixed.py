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
    page_title="Traffic Intersection Monitoring",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    /* Modern color palette */
    :root {
        --primary-color: #4F46E5;
        --primary-light: #818CF8;
        --primary-dark: #3730A3;
        --secondary-color: #10B981;
        --background-light: #F8FAFC;
        --text-dark: #1E293B;
        --text-light: #64748B;
        --white: #FFFFFF;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.12);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
    }

    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: var(--background-light);
    }
    
    /* Title styling */
    .title {
        color: var(--primary-color);
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #4F46E5 0%, #818CF8 100%);
        color: white;
        border-radius: 16px;
        box-shadow: var(--shadow-lg);
    }
    
    /* Upload box styling */
    .upload-box {
        padding: 2rem;
        border: 2px dashed var(--primary-light);
        border-radius: 16px;
        background-color: var(--white);
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: var(--primary-color);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }

    .upload-box h3 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .upload-box p {
        color: var(--text-light);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: var(--white);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    
    /* Instructions card styling */
    .instructions {
        background-color: var(--white);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
    }

    .instructions h3 {
        color: var(--primary-color);
        margin-bottom: 1rem;
    }
    
    /* Feature list styling */
    .feature-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .feature-list li {
        margin-bottom: 1rem;
        padding-left: 1.5rem;
        position: relative;
        color: var(--text-dark);
    }
    
    .feature-list li:before {
        content: "→";
        color: var(--primary-color);
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    /* Info button styling */
    .info-button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        margin-left: 8px;
        font-size: 16px;
        border: none;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }

    .info-button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .info-content {
        background-color: var(--white);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
        margin-top: 1rem;
        border-left: 4px solid var(--primary-color);
    }

    /* Select box styling */
    .stSelectbox {
        border-radius: 12px;
    }

    /* DataFrame styling */
    .dataframe {
        border: none !important;
        box-shadow: var(--shadow-sm);
        border-radius: 12px;
    }

    /* Progress bar styling */
    .stProgress > div > div {
        background-color: var(--primary-light);
    }

    /* Success message styling */
    .success {
        background-color: var(--secondary-color);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 1rem 0;
        text-align: center;
        font-size: 0.875rem;
        box-shadow: var(--shadow-lg);
        z-index: 100;
    }

    .footer-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }

    .footer-links a {
        color: white;
        text-decoration: none;
        margin: 0 1rem;
        transition: all 0.3s ease;
    }

    .footer-links a:hover {
        color: var(--primary-light);
    }

    /* Add padding to main content to prevent footer overlap */
    .main {
        padding-bottom: 5rem;
    }

    /* Header container styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        background: var(--white);
        border-radius: 16px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
    }

    .logo-container {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .logo-image {
        width: 40px;  /* Adjust size as needed */
        height: auto;
        object-fit: contain;
    }

    .logo-text {
        color: var(--primary-color);
        font-size: 1.5rem;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Navigation styling */
    .nav-links {
        display: flex;
        gap: 2rem;
    }

    .nav-link {
        color: var(--text-dark);
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .nav-link:hover {
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# Load YOLO model for car detection
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')
    return model

# Helper function to calculate the center of the bounding box
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# Determine the car movement direction at the intersection
def calculate_movement_direction(previous_center, current_center, camera_position):
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    
    if camera_position == "north":  # Example: Camera facing south
        if dy > 0:
            return "STRAIGHT (southbound)"
        elif dx > 0:
            return "RIGHT (westbound)"
        elif dx < 0:
            return "LEFT (eastbound)"
        else:
            return "UNKNOWN"
    elif camera_position == "south":  # Example: Camera facing north
        if dy < 0:
            return "STRAIGHT (northbound)"
        elif dx > 0:
            return "LEFT (westbound)"
        elif dx < 0:
            return "RIGHT (eastbound)"
        else:
            return "UNKNOWN"
    elif camera_position == "east":  # Example: Camera facing west
        if dx < 0:
            return "STRAIGHT (westbound)"
        elif dy > 0:
            return "LEFT (southbound)"
        elif dy < 0:
            return "RIGHT (northbound)"
        else:
            return "UNKNOWN"
    elif camera_position == "west":  # Example: Camera facing east
        if dx > 0:
            return "STRAIGHT (eastbound)"
        elif dy > 0:
            return "RIGHT (southbound)"
        elif dy < 0:
            return "LEFT (northbound)"
        else:
            return "UNKNOWN"
    else:
        return "UNKNOWN"

# Track cars across frames and classify movements
def track_cars(detections, previous_tracks, frame_id, camera_position, max_distance=50):
    current_tracks = {}
    used_previous = set()

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
            dist = np.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)

            if dist < min_dist and dist < max_distance:
                min_dist = dist
                best_match = track_id

        if best_match is not None:
            # Update existing track with movement direction
            movement = calculate_movement_direction(get_center(previous_tracks[best_match]['bbox']), center, camera_position)
            current_tracks[best_match] = {
                'bbox': bbox,
                'center': center,
                'last_seen': frame_id,
                'movement': movement
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

def process_video(video_path, camera_position):
    model = load_model()
    cap = cv2.VideoCapture(video_path)

    tracks = {}
    car_movements = defaultdict(list)
    frame_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

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

        # Run YOLOv8 detection
        results = model(frame)

        # Filter car detections (class 2 in COCO)
        detections = []
        for r in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if class_id == 2 and score > 0.5:  # Car class with confidence threshold
                detections.append([x1, y1, x2, y2])

        # Track cars and record movements
        tracks = track_cars(detections, tracks, frame_count, camera_position)

        for track_id, track_info in tracks.items():
            car_movements[track_id].append({
                'frame': frame_count,
                'movement': track_info['movement'],
                'timestamp': time.time()
            })

    cap.release()
    progress_bar.empty()
    status_text.empty()

    # Summarize movements
    final_movements = []
    for car_id, movements in car_movements.items():
        if len(movements) > 5:  # Consider only cars tracked for more than 5 frames
            most_common_movement = max(set(m['movement'] for m in movements),
                                       key=lambda x: sum(1 for m in movements if m['movement'] == x))
            final_movements.append({
                'car_id': car_id,
                'movement': most_common_movement,
                'timestamp': movements[-1]['timestamp']
            })

    return final_movements

def create_info_button(key, title, content):
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
    with col2:
        show_info = st.button("ℹ️", key=f"info_button_{key}")
    
    if show_info:
        st.session_state[f'show_{key}'] = not st.session_state.get(f'show_{key}', False)
    
    if st.session_state.get(f'show_{key}', False):
        st.markdown(f"""
            <div class="info-content">
                {content}
            </div>
        """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
        <div class="footer">
            <div class="footer-content">
                <div class="footer-copyright">
                    © 2024 VidEye AI. All rights reserved.
                </div>
                <div class="footer-company">
                    Powered by Advanced Computer Vision Technology
                </div>
                <div class="footer-links">
                    <a href="https://videye.ai" target="_blank">🌐 Website</a>
                    <a href="https://linkedin.com/company/videye-ai" target="_blank">💼 LinkedIn</a>
                    <a href="mailto:contact@videye.ai">✉️ Contact</a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def add_logo_and_title():
    try:
        # Get the logo file path
        logo_path = "logo1.png"  # Logo in base folder
        logo_base64 = get_base64_of_bin_file(logo_path)
        
        st.markdown(f"""
            <div class="header-container">
                <div class="logo-container">
                    <img src="data:image/png;base64,{logo_base64}" class="logo-image" alt="VidEye AI Logo"/>
                    <span class="logo-text">VidEye AI</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        # Fallback to text if logo loading fails
        st.markdown("""
            <div class="header-container">
                <div class="logo-container">
                    <span style="font-size: 32px; margin-right: 10px;">🎥</span>
                    <span class="logo-text">VidEye AI</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

def main():
    # Add logo and title
    add_logo_and_title()
    
    # Main title
    st.markdown("<h1 class='title'>🚦 Traffic Intersection Monitoring System</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <div class='upload-box'>
                <h3>📤 Upload Video</h3>
                <p>Support formats: MP4, AVI</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['mp4', 'avi'])
        
        # Instructions info button
        instructions_content = """
            <h4>📝 How to Use</h4>
            <ol>
                <li><strong>Upload Video:</strong> Select an MP4 or AVI format traffic video</li>
                <li><strong>Select Position:</strong> Choose the camera's viewing direction</li>
                <li><strong>Process:</strong> Click 'Process Video' and wait for analysis</li>
                <li><strong>Review:</strong> Examine the results and download if needed</li>
            </ol>
            <p><em>Note: Processing time depends on video length and complexity</em></p>
        """
        create_info_button("instructions", "Instructions", instructions_content)

        camera_position = st.selectbox(
            "📹 Select Camera Position",
            ['north', 'south', 'east', 'west'],
            format_func=lambda x: x.title()
        )

        if uploaded_file is not None:
            if st.button("🎯 Process Video"):
                with st.spinner("🔄 Processing video..."):
                    movements = process_video(uploaded_file.name, camera_position)

                    # Convert to DataFrame
                    df_movements = pd.DataFrame(movements)

                    # Display results
                    st.success("✅ Video processing completed!")

                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.subheader("📊 Analysis Summary")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4 style='margin: 0; font-size: 1rem;'>Total Cars</h4>
                                <h2 style='margin: 0.5rem 0; font-size: 2rem;'>{len(df_movements)}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    with metrics_col2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4 style='margin: 0; font-size: 1rem;'>Left Turns</h4>
                                <h2 style='margin: 0.5rem 0; font-size: 2rem;'>{len(df_movements[df_movements['movement'].str.contains('LEFT')])}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    with metrics_col3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4 style='margin: 0; font-size: 1rem;'>Right Turns</h4>
                                <h2 style='margin: 0.5rem 0; font-size: 2rem;'>{len(df_movements[df_movements['movement'].str.contains('RIGHT')])}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Display movement breakdown
                    st.subheader("🔄 Movement Breakdown")
                    movement_counts = df_movements['movement'].value_counts()
                    st.bar_chart(movement_counts)

                    # Display detailed results
                    st.subheader("📋 Detailed Results")
                    st.dataframe(df_movements, use_container_width=True)

                    # Export option
                    csv = df_movements.to_csv(index=False)
                    st.download_button(
                        label="📥 Download results as CSV",
                        data=csv,
                        file_name="car_movements.csv",
                        mime="text/csv"
                    )

    with col2:
        st.markdown("""
        <div class='instructions'>
            <h3>🎯 Key Features</h3>
            <ul class='feature-list'>
                <li>Advanced YOLOv8 Detection</li>
                <li>Real-time Movement Tracking</li>
                <li>Multi-directional Analysis</li>
                <li>Detailed Statistics</li>
                <li>CSV Export Options</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Add footer at the end of main
    add_footer()

if __name__ == "__main__":
    # Initialize session state for instructions info button only
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = False
    
    main()
