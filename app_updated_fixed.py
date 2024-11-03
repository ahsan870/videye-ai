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
# Determine the car movement direction at the intersection
def calculate_movement_direction(previous_center, current_center, camera_position):
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    
    if camera_position == "north":  # Camera facing south
        if dy > 0:
            return "STRAIGHT (southbound)"
        elif dx > 0:
            return "RIGHT (westbound)"
        elif dx < 0:
            return "LEFT (eastbound)"
        else:
            return "TOWARD CAMERA"
    elif camera_position == "south":  # Camera facing north
        if dy < 0:
            return "STRAIGHT (northbound)"
        elif dx > 0:
            return "LEFT (westbound)"
        elif dx < 0:
            return "RIGHT (eastbound)"
        else:
            return "TOWARD CAMERA"
    elif camera_position == "east":  # Camera facing west
        if dx < 0:
            return "STRAIGHT (westbound)"
        elif dy > 0:
            return "LEFT (southbound)"
        elif dy < 0:
            return "RIGHT (northbound)"
        else:
            return "TOWARD CAMERA"
    elif camera_position == "west":  # Camera facing east
        if dx > 0:
            return "STRAIGHT (eastbound)"
        elif dy > 0:
            return "RIGHT (southbound)"
        elif dy < 0:
            return "LEFT (northbound)"
        else:
            return "TOWARD CAMERA"
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

def main():
    st.markdown("<h1 class='title'>🚦 Traffic Intersection Monitoring</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi'])
        st.markdown("</div>", unsafe_allow_html=True)

        camera_position = st.selectbox("Select Camera Position", ['north', 'south', 'east', 'west'])

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            # Store processing state in session state
            if 'processing_complete' not in st.session_state:
                st.session_state.processing_complete = False

            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    movements = process_video(tfile.name, camera_position)
                    df_movements = pd.DataFrame(movements)
                    st.session_state.processing_complete = True
                    st.session_state.df_movements = df_movements  # Store results in session state

            # Show analysis only after processing is complete
            if st.session_state.get('processing_complete', False):
                df_movements = st.session_state.df_movements
                
                # Display results
                st.success("Video processing completed!")

                # Total Cars - Separate and prominent
                st.metric("🚗 Total Vehicles Detected", len(df_movements))
                
                st.subheader("📊 Movement Analysis")
                
                # First row of metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    straight_count = len(df_movements[df_movements['movement'].str.contains('STRAIGHT')])
                    st.metric("⬆️ Straight", straight_count)
                with col2:
                    left_turns = len(df_movements[df_movements['movement'].str.contains('LEFT')])
                    st.metric("↪️ Left", left_turns)
                with col3:
                    right_turns = len(df_movements[df_movements['movement'].str.contains('RIGHT')])
                    st.metric("↩️ Right", right_turns)
                with col4:
                    toward_count = len(df_movements[df_movements['movement'].str.contains('TOWARD')])
                    st.metric("🎯 Toward", toward_count)

                # Second row - Direction Analysis
                st.subheader("🔄 Direction Analysis")
                dir_col1, dir_col2, dir_col3, dir_col4 = st.columns(4)
                
                with dir_col1:
                    northbound = len(df_movements[df_movements['movement'].str.contains('northbound')])
                    st.metric("⬆️ North", northbound)
                with dir_col2:
                    southbound = len(df_movements[df_movements['movement'].str.contains('southbound')])
                    st.metric("⬇️ South", southbound)
                with dir_col3:
                    eastbound = len(df_movements[df_movements['movement'].str.contains('eastbound')])
                    st.metric("➡️ East", eastbound)
                with dir_col4:
                    westbound = len(df_movements[df_movements['movement'].str.contains('westbound')])
                    st.metric("⬅️ West", westbound)

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

    # Show instructions and features only when not processing or showing results
    with col2:
        if not st.session_state.get('processing_complete', False):
            st.markdown('''
            ### 📝 Instructions
            1. Upload a video file (MP4 or AVI format)
            2. Select the camera position (north, south, east, or west)
            3. Click 'Process Video' button
            4. View results and download the report

            ### 🎯 Features
            - Car detection using YOLOv8
            - Movement tracking (left, right, straight, toward/away from the camera)
            - Detailed statistics
            - Downloadable reports
            ''')

if __name__ == "__main__":
    # Initialize session state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    main()
