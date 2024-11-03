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
def calculate_movement_direction(previous_center, current_center, camera_position, road_type='4-way intersection'):
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    
    if road_type == '2-way road':
        # Simplified logic for 2-way roads
        if camera_position == "north":  # Camera facing south
            if dy > 0:
                return "STRAIGHT (southbound)"
            elif dy < 0:
                return "STRAIGHT (northbound)"
            else:
                return "STATIONARY"
        elif camera_position == "south":  # Camera facing north
            if dy < 0:
                return "STRAIGHT (northbound)"
            elif dy > 0:
                return "STRAIGHT (southbound)"
            else:
                return "STATIONARY"
    
    elif road_type == 'T-junction':
        # Modified logic for T-junctions
        if camera_position == "north":  # Camera facing south
            if dy > 0:
                return "STRAIGHT (southbound)"
            elif dx > 0:
                return "RIGHT (westbound)"
            elif dx < 0:
                return "LEFT (eastbound)"
            else:
                return "STATIONARY"
        elif camera_position == "south":  # Camera facing north
            if dy < 0:
                return "STRAIGHT (northbound)"
            elif dx > 0:
                return "RIGHT (eastbound)"
            elif dx < 0:
                return "LEFT (westbound)"
            else:
                return "STATIONARY"
        elif camera_position == "east":  # Camera facing west
            if dx < 0:
                return "STRAIGHT (westbound)"
            elif dy > 0:
                return "RIGHT (southbound)"
            elif dy < 0:
                return "LEFT (northbound)"
            else:
                return "STATIONARY"
    
    else:  # 4-way intersection
        if camera_position == "north":
            if abs(dy) > abs(dx):
                if dy > 0:
                    return "STRAIGHT (southbound)"
                else:
                    return "STRAIGHT (northbound)"
            else:
                if dx > 0:
                    return "RIGHT (westbound)"
                else:
                    return "LEFT (eastbound)"
        elif camera_position == "south":
            if abs(dy) > abs(dx):
                if dy < 0:
                    return "STRAIGHT (northbound)"
                else:
                    return "STRAIGHT (southbound)"
            else:
                if dx > 0:
                    return "RIGHT (eastbound)"
                else:
                    return "LEFT (westbound)"
        elif camera_position == "east":
            if abs(dx) > abs(dy):
                if dx < 0:
                    return "STRAIGHT (westbound)"
                else:
                    return "STRAIGHT (eastbound)"
            else:
                if dy > 0:
                    return "RIGHT (southbound)"
                else:
                    return "LEFT (northbound)"
        elif camera_position == "west":
            if abs(dx) > abs(dy):
                if dx > 0:
                    return "STRAIGHT (eastbound)"
                else:
                    return "STRAIGHT (westbound)"
            else:
                if dy > 0:
                    return "RIGHT (northbound)"
                else:
                    return "LEFT (southbound)"
        
        return "UNKNOWN"

    return "UNKNOWN"

# Track cars across frames and classify movements
def track_cars(detections, previous_tracks, frame_id, camera_position, road_type):
    max_distance = 50
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

def process_video(video_path, camera_position, road_type):
    # Add validation
    if road_type == '2-way road' and camera_position not in ['north', 'south']:
        st.error("Invalid camera position for 2-way road")
        return None
    
    if road_type == 'T-junction' and camera_position == 'west':
        st.error("Invalid camera position for T-junction")
        return None
    
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
        tracks = track_cars(detections, tracks, frame_count, camera_position, road_type)

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

# Add this to your CSS
st.markdown("""
<style>
    /* ... (previous CSS) ... */

    /* Instructions Modal */
    .instructions-modal {
        position: fixed;
        right: 20px;
        top: 80px;
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        max-width: 350px;
        transition: all 0.3s ease;
    }

    .instructions-modal h3 {
        color: var(--primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .instructions-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .instructions-list li {
        margin-bottom: 0.75rem;
        padding-left: 1.5rem;
        position: relative;
    }

    .instructions-list li:before {
        content: '→';
        position: absolute;
        left: 0;
        color: var(--primary);
    }

    /* Help Button */
    .help-button {
        position: fixed;
        right: 20px;
        top: 20px;
        background: var(--primary);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: var(--shadow);
        z-index: 1001;
        transition: all 0.3s ease;
    }

    .help-button:hover {
        background: var(--secondary);
        transform: scale(1.05);
    }

    /* Navigation styles */
    .nav-container {
        background: white;
        padding: 1rem 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .nav-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        text-decoration: none;
    }

    .nav-logo img {
        height: 40px;
        width: auto;
    }

    .nav-logo span {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary);
    }

    .nav-links {
        display: flex;
        gap: 2rem;
    }

    .nav-link {
        color: var(--text-dark);
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        transition: all 0.2s ease;
    }

    .nav-link:hover {
        background: var(--background);
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# Add this to your session state initialization
if 'show_instructions' not in st.session_state:
    st.session_state.show_instructions = False

# Add this function to handle the toggle
def toggle_instructions():
    st.session_state.show_instructions = not st.session_state.show_instructions

# Add this to your main() function, right after add_nav():
def show_help_button():
    # Help button
    st.markdown("""
        <div class="help-button" onclick="document.getElementById('instructions').style.display = 
            document.getElementById('instructions').style.display === 'none' ? 'block' : 'none';">
            <span style="font-size: 1.2rem;">?</span>
        </div>
    """, unsafe_allow_html=True)

    # Instructions modal
    display = "block" if st.session_state.show_instructions else "none"
    st.markdown(f"""
        <div id="instructions" class="instructions-modal" style="display: {display};">
            <h3>📝 Instructions</h3>
            <ul class="instructions-list">
                <li>Upload a video file (MP4 or AVI format)</li>
                <li>Select the road type from the dropdown</li>
                <li>Choose the camera position</li>
                <li>Click 'Process Video' button</li>
                <li>View analysis results and download report</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Function to load and encode the logo
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return ""

# Navigation function
def add_nav():
    try:
        # Try to load logo
        logo_path = "logo1.png"  # Make sure this path is correct
        logo_base64 = get_base64_of_bin_file(logo_path)
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="logo-image" alt="VidEye AI"/>'
    except:
        # Fallback to text if logo fails to load
        logo_html = '🚦'

    st.markdown(f"""
        <div class="nav-container">
            <div class="nav-logo">
                {logo_html}
                <span>VidEye AI</span>
            </div>
            <div class="nav-links">
                <a href="/" class="nav-link">Home</a>
                <a href="/about" class="nav-link">About</a>
                <a href="/contact" class="nav-link">Contact</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    add_nav()
    show_help_button()  # Add this line
    
    # Main content container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Title section
    st.markdown("""
        <div class="title-section">
            <h1>🚦 Traffic Intersection Monitoring</h1>
            <p>Advanced AI-powered traffic analysis for smarter urban planning</p>
        </div>
    """, unsafe_allow_html=True)

    # Add JavaScript for toggle functionality
    st.markdown("""
        <script>
            const toggleInstructions = () => {
                const instructions = document.getElementById('instructions');
                instructions.style.display = instructions.style.display === 'none' ? 'block' : 'none';
            }
        </script>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi'])
        st.markdown("</div>", unsafe_allow_html=True)

        # Update camera position selection with road type
        road_type = st.selectbox("Select Road Type", ['4-way intersection', '2-way road', 'T-junction'])
        
        # Show appropriate camera positions based on road type
        if road_type == '2-way road':
            camera_position = st.selectbox("Select Camera Position", ['north', 'south'])
        elif road_type == 'T-junction':
            camera_position = st.selectbox("Select Camera Position", ['north', 'south', 'east'])
        else:  # 4-way intersection
            camera_position = st.selectbox("Select Camera Position", ['north', 'south', 'east', 'west'])

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            # Store processing state in session state
            if 'processing_complete' not in st.session_state:
                st.session_state.processing_complete = False

            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    movements = process_video(tfile.name, camera_position, road_type)
                    df_movements = pd.DataFrame(movements)
                    st.session_state.processing_complete = True
                    st.session_state.df_movements = df_movements  # Store results in session state

            # Show analysis only after processing is complete
            if st.session_state.get('processing_complete', False):
                df_movements = st.session_state.df_movements
                
                # Display results
                st.success("Video processing completed!")

                # Total Cars - Always show
                st.metric("🚗 Total Vehicles Detected", len(df_movements))
                
                if road_type == '2-way road':
                    # Movement Analysis for 2-way road
                    st.subheader("📊 Movement Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        northbound = len(df_movements[df_movements['movement'].str.contains('northbound')])
                        st.metric("⬆️ Northbound", northbound)
                    with col2:
                        southbound = len(df_movements[df_movements['movement'].str.contains('southbound')])
                        st.metric("⬇️ Southbound", southbound)
                        
                    # Percentage Analysis
                    total = northbound + southbound
                    if total > 0:
                        st.subheader("🔄 Direction Distribution")
                        dir_col1, dir_col2 = st.columns(2)
                        with dir_col1:
                            st.metric("⬆️ Northbound %", f"{(northbound/total)*100:.1f}%")
                        with dir_col2:
                            st.metric("⬇️ Southbound %", f"{(southbound/total)*100:.1f}%")
                
                elif road_type == 'T-junction':
                    # Movement Analysis for T-junction
                    st.subheader("📊 Movement Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        straight = len(df_movements[df_movements['movement'].str.contains('STRAIGHT')])
                        st.metric("⬆️ Straight", straight)
                    with col2:
                        left = len(df_movements[df_movements['movement'].str.contains('LEFT')])
                        st.metric("↩️ Left Turn", left)
                    with col3:
                        right = len(df_movements[df_movements['movement'].str.contains('RIGHT')])
                        st.metric("↪️ Right Turn", right)
                        
                    # Direction Analysis for T-junction
                    st.subheader("🔄 Direction Analysis")
                    dir_col1, dir_col2, dir_col3 = st.columns(3)
                    
                    if camera_position == "north":
                        with dir_col1:
                            southbound = len(df_movements[df_movements['movement'].str.contains('southbound')])
                            st.metric("⬇️ Southbound", southbound)
                        with dir_col2:
                            eastbound = len(df_movements[df_movements['movement'].str.contains('eastbound')])
                            st.metric("➡️ Eastbound", eastbound)
                        with dir_col3:
                            westbound = len(df_movements[df_movements['movement'].str.contains('westbound')])
                            st.metric("⬅️ Westbound", westbound)
                    elif camera_position == "south":
                        with dir_col1:
                            northbound = len(df_movements[df_movements['movement'].str.contains('northbound')])
                            st.metric("⬆️ Northbound", northbound)
                        with dir_col2:
                            eastbound = len(df_movements[df_movements['movement'].str.contains('eastbound')])
                            st.metric("➡️ Eastbound", eastbound)
                        with dir_col3:
                            westbound = len(df_movements[df_movements['movement'].str.contains('westbound')])
                            st.metric("⬅️ Westbound", westbound)
                
                else:  # 4-way intersection
                    # Movement Analysis
                    st.subheader("📊 Movement Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        straight = len(df_movements[df_movements['movement'].str.contains('STRAIGHT')])
                        st.metric("⬆️ Straight", straight)
                    with col2:
                        left = len(df_movements[df_movements['movement'].str.contains('LEFT')])
                        st.metric("↩️ Left", left)
                    with col3:
                        right = len(df_movements[df_movements['movement'].str.contains('RIGHT')])
                        st.metric("↪️ Right", right)
                    with col4:
                        stationary = len(df_movements[df_movements['movement'].str.contains('STATIONARY')])
                        st.metric("🚫 Stationary", stationary)

                    # Direction Analysis
                    st.subheader("🔄 Direction Analysis")
                    dir_col1, dir_col2, dir_col3, dir_col4 = st.columns(4)
                    
                    with dir_col1:
                        northbound = len(df_movements[df_movements['movement'].str.contains('northbound')])
                        st.metric("⬆️ Northbound", northbound)
                    with dir_col2:
                        southbound = len(df_movements[df_movements['movement'].str.contains('southbound')])
                        st.metric("⬇️ Southbound", southbound)
                    with dir_col3:
                        eastbound = len(df_movements[df_movements['movement'].str.contains('eastbound')])
                        st.metric("➡️ Eastbound", eastbound)
                    with dir_col4:
                        westbound = len(df_movements[df_movements['movement'].str.contains('westbound')])
                        st.metric("⬅️ Westbound", westbound)

                # Display movement breakdown chart
                st.subheader("📊 Movement Distribution")
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

            # Clean up temporary file
            Path(tfile.name).unlink()

if __name__ == "__main__":
    # Initialize session state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    main()
