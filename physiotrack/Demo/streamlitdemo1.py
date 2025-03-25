#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit demo for the Range of Motion (ROM) analysis functionality of PhysioTrack.
This is an interactive web app that allows users to upload a video file, process it,
and analyze the ROM.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Import PhysioTrack
from physiotrack import PhysioTrack

# Set page config
st.set_page_config(
    page_title="PhysioTrack ROM Analysis",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a4a4a;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 1.2rem;
        font-weight: bold;
        color: #555;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0066cc;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frames = []
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frames.append(img.copy())
        return img
    
    def get_frames(self):
        return self.frames

def autoplay_video(video_path):
    """Autoplay the video"""
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

def get_video_duration(file_path):
    """Get video duration using OpenCV"""
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration

def save_frames_to_video(frames, output_path, fps=30):
    """Save frames to a video file"""
    if not frames:
        return None
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path

def load_mot_file(mot_file):
    """Load the angles MOT file"""
    # Skip the header lines
    with open(mot_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                header_rows = i
                break
    
    # Read the data
    angles_data = pd.read_csv(mot_file, sep='\t', skiprows=header_rows)
    return angles_data

def analyze_rom(angles_data):
    """Calculate ROM for each joint"""
    rom_results = {}
    for col in angles_data.columns:
        if col == 'time':
            continue
        
        # Calculate min, max, and ROM
        min_val = angles_data[col].min()
        max_val = angles_data[col].max()
        rom = max_val - min_val
        mean_val = angles_data[col].mean()
        std_val = angles_data[col].std()
        
        rom_results[col] = {
            'min': min_val,
            'max': max_val,
            'rom': rom,
            'mean': mean_val,
            'std': std_val
        }
    
    return rom_results

def plot_rom_results(rom_results, angles_data):
    """Plot ROM results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ROM bar chart
    joints = list(rom_results.keys())
    rom_values = [data['rom'] for data in rom_results.values()]
    bars = ax1.bar(joints, rom_values, color='skyblue')
    
    # Add data labels
    for bar, value in zip(bars, rom_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}°',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    ax1.set_title('Range of Motion Analysis', fontsize=14)
    ax1.set_ylabel('ROM (degrees)', fontsize=12)
    ax1.set_ylim(0, max(rom_values) * 1.2)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Time series plot for key joints
    key_joints = ['Right knee', 'Left knee', 'Right hip', 'Left hip']
    for col in key_joints:
        if col in angles_data.columns:
            ax2.plot(angles_data['time'], angles_data[col], label=col, linewidth=2)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Angle (degrees)', fontsize=12)
    ax2.set_title('Joint Angle Time Series', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_heatmap(angles_data):
    """Create a heatmap of joint angles over time"""
    # Exclude time column
    data_for_heatmap = angles_data.drop(columns=['time'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize data for better visualization
    normalized_data = (data_for_heatmap - data_for_heatmap.min()) / (data_for_heatmap.max() - data_for_heatmap.min())
    
    # Create meshgrid for heatmap
    x = angles_data['time']
    y = np.arange(len(data_for_heatmap.columns))
    X, Y = np.meshgrid(x, y)
    
    # Plot heatmap
    c = ax.pcolormesh(X, Y, normalized_data.T, cmap='viridis', shading='auto')
    
    # Set labels
    ax.set_yticks(np.arange(len(data_for_heatmap.columns)) + 0.5)
    ax.set_yticklabels(data_for_heatmap.columns)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Joint Angle Heatmap', fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Normalized Angle Value', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown("<h1 class='main-header'>PhysioTrack ROM Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Analyze range of motion from video recordings</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/150x150.png?text=PhysioTrack", use_column_width=True)
    st.sidebar.title("Settings")
    
    # Input methods
    input_method = st.sidebar.radio("Input Method:", ["Upload Video", "Use Webcam", "Use Demo Video"])
    
    # Model settings
    model_settings = st.sidebar.expander("Model Settings", expanded=False)
    with model_settings:
        pose_model = model_settings.selectbox("Pose Model", ["body_with_feet", "whole_body", "body"], index=0)
        model_mode = model_settings.selectbox("Mode", ["lightweight", "balanced", "performance"], index=1)
        det_frequency = model_settings.slider("Detection Frequency", 1, 10, 4, 
                                          help="Run detection every N frames. Higher values = faster but less accurate.")
    
    # Analysis settings
    analysis_settings = st.sidebar.expander("Analysis Settings", expanded=False)
    with analysis_settings:
        person_height = analysis_settings.slider("Person Height (m)", 1.0, 2.2, 1.70, 0.05)
        multiperson = analysis_settings.checkbox("Multiple Persons", value=False)
        show_realtime = analysis_settings.checkbox("Show Realtime Results", value=True)
        to_meters = analysis_settings.checkbox("Convert to Meters", value=True)
        
        joint_angles = analysis_settings.multiselect(
            "Joint Angles to Analyze",
            ["Right ankle", "Left ankle", "Right knee", "Left knee", "Right hip", "Left hip", 
             "Right shoulder", "Left shoulder", "Right elbow", "Left elbow"],
            default=["Right knee", "Left knee", "Right hip", "Left hip", "Right shoulder", "Left shoulder"]
        )
        
        segment_angles = analysis_settings.multiselect(
            "Segment Angles to Analyze",
            ["Right foot", "Left foot", "Right shank", "Left shank", "Right thigh", "Left thigh", 
             "Pelvis", "Trunk", "Shoulders", "Head"],
            default=["Right thigh", "Left thigh", "Trunk"]
        )
        
        display_options = analysis_settings.multiselect(
            "Display Angle Values On",
            ["body", "list"],
            default=["body", "list"]
        )
    
    # Processing settings
    processing_settings = st.sidebar.expander("Processing Settings", expanded=False)
    with processing_settings:
        interpolate = processing_settings.checkbox("Interpolate Missing Data", value=True)
        filter_data = processing_settings.checkbox("Filter Data", value=True)
        filter_type = processing_settings.selectbox(
            "Filter Type", 
            ["butterworth", "gaussian", "loess", "median"],
            index=0
        )
    
    # Main content area
    if input_method == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "m4v"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.success("Video uploaded successfully!")
            
            # Show video preview
            st.subheader("Video Preview")
            autoplay_video(video_path)
            
            # Process button
            if st.button("Process Video"):
                process_video(video_path, pose_model, model_mode, det_frequency, 
                             person_height, multiperson, show_realtime, to_meters,
                             joint_angles, segment_angles, display_options,
                             interpolate, filter_data, filter_type)
    
    elif input_method == "Use Webcam":
        st.subheader("Webcam Capture")
        
        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if st.button("Capture and Process"):
            if ctx.video_transformer:
                frames = ctx.video_transformer.get_frames()
                if frames:
                    # Save frames to temporary video file
                    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    saved_path = save_frames_to_video(frames, temp_video_path)
                    
                    if saved_path:
                        st.success(f"Captured {len(frames)} frames!")
                        # Process the video
                        process_video(saved_path, pose_model, model_mode, det_frequency, 
                                     person_height, multiperson, show_realtime, to_meters,
                                     joint_angles, segment_angles, display_options,
                                     interpolate, filter_data, filter_type)
                else:
                    st.error("No frames captured. Please allow webcam access and try again.")
            else:
                st.error("WebRTC connection not established. Please start the webcam stream.")
    
    elif input_method == "Use Demo Video":
        st.subheader("Demo Video")
        
        # Path to demo video (you should include this in your package)
        demo_dir = Path(__file__).resolve().parent
        demo_video = demo_dir / "demo.mp4"
        
        if not demo_video.exists():
            st.error(f"Demo video not found at {demo_video}")
            demo_video = None
        else:
            st.success(f"Using demo video: {demo_video}")
            autoplay_video(str(demo_video))
            
            if st.button("Process Demo Video"):
                process_video(demo_video, pose_model, model_mode, det_frequency, 
                             person_height, multiperson, show_realtime, to_meters,
                             joint_angles, segment_angles, display_options,
                             interpolate, filter_data, filter_type)
    
    # Footer
    st.markdown("<div class='footer'>PhysioTrack - Range of Motion Analysis Tool</div>", unsafe_allow_html=True)

def process_video(video_path, pose_model, model_mode, det_frequency, 
                 person_height, multiperson, show_realtime, to_meters,
                 joint_angles, segment_angles, display_options,
                 interpolate, filter_data, filter_type):
    """Process the video and display results"""
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Setting up analysis parameters...")
    
    # Create output directory
    results_dir = Path(tempfile.mkdtemp())
    
    # Create configuration dictionary
    config_dict = {
        'project': {
            'video_input': [str(video_path)],
            'px_to_m_person_height': float(person_height),
            'visible_side': ['auto'],
            'time_range': [],
            'load_trc_px': '',
            'compare': False
        },
        'process': {
            'multiperson': multiperson,
            'show_realtime_results': show_realtime,
            'save_vid': True,
            'save_img': False,
            'save_pose': True,
            'calculate_angles': True,
            'save_angles': True,
            'result_dir': str(results_dir)
        },
        'pose': {
            'pose_model': pose_model,
            'mode': model_mode,
            'det_frequency': det_frequency,
            'device': 'auto',
            'backend': 'auto',
            'tracking_mode': 'physiotrack',
            'deepsort_params': """{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, 'max_cosine_distance':0.3, 'nn_budget':200, 'max_iou_distance':0.8, 'embedder_gpu': True}""",
            'keypoint_likelihood_threshold': 0.3,
            'average_likelihood_threshold': 0.5,
            'keypoint_number_threshold': 0.3,
            'slowmo_factor': 1
        },
        'angles': {
            'joint_angles': joint_angles,
            'segment_angles': segment_angles,
            'display_angle_values_on': display_options,
            'fontSize': 0.3,
            'flip_left_right': True,
            'correct_segment_angles_with_floor_angle': True
        },
        'px_to_meters_conversion': {
            'to_meters': to_meters,
            'make_c3d': True,
            'calib_file': '',
            'floor_angle': 'auto',
            'xy_origin': ['auto'],
            'save_calib': True
        },
        'post-processing': {
            'interpolate': interpolate,
            'filter': filter_data,
            'show_graphs': False,
            'filter_type': filter_type,
            'butterworth': {'order': 4, 'cut_off_frequency': 6},
            'gaussian': {'sigma_kernel': 1},
            'loess': {'nb_values_used': 5},
            'median': {'kernel_size': 3},
            'interp_gap_smaller_than': 10,
            'fill_large_gaps_with': 'last_value'
        },
        'kinematics': {
            'do_ik': False,
            'use_augmentation': False,
            'use_contacts_muscles': True,
            'participant_mass': [67.0],
            'right_left_symmetry': True,
            'default_height': float(person_height),
            'remove_individual_scaling_setup': True,
            'remove_individual_ik_setup': True,
            'fastest_frames_to_remove_percent': 0.1,
            'close_to_zero_speed_px': 50,
            'close_to_zero_speed_m': 0.2,
            'large_hip_knee_angles': 45,
            'trimmed_extrema_percent': 0.5,
            'osim_setup_path': '../OpenSim_setup'
        },
        'logging': {
            'use_custom_logging': True
        }
    }
    
    # Run PhysioTrack with the configuration
    status_text.text("Processing video...")
    progress_bar.progress(10)
    
    try:
        PhysioTrack.process(config_dict)
        progress_bar.progress(70)
        status_text.text("Processing complete. Analyzing ROM...")
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return
    
    # Find the angles.mot file
    mot_files = list(results_dir.glob("*_angles*.mot"))
    
    if not mot_files:
        st.error("No angle files found. Processing may have failed.")
        return
    
    # Load the processed video
    processed_videos = list(results_dir.glob("*.mp4"))
    if processed_videos:
        st.subheader("Processed Video")
        autoplay_video(str(processed_videos[0]))
    
    # Load the angles file
    mot_file = mot_files[0]
    angles_data = load_mot_file(mot_file)
    
    progress_bar.progress(80)
    status_text.text("Calculating range of motion...")
    
    # Calculate ROM for each joint
    rom_results = analyze_rom(angles_data)
    
    progress_bar.progress(90)
    status_text.text("Generating visualizations...")
    
    # Display results
    st.subheader("Range of Motion Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary", "Charts", "Data"])
    
    with tab1:
        # Summary metrics
        st.markdown("<h3>Key Range of Motion Metrics</h3>", unsafe_allow_html=True)
        
        # Display ROM metrics in a grid
        cols = st.columns(3)
        for i, (joint, data) in enumerate(rom_results.items()):
            with cols[i % 3]:
                st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-label'>{joint}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{data['rom']:.1f}°</p>", unsafe_allow_html=True)
                st.markdown(f"<p>Min: {data['min']:.1f}° | Max: {data['max']:.1f}°</p>", unsafe_allow_html=True)
                st.markdown(f"<p>Mean: {data['mean']:.1f}° ± {data['std']:.1f}°</p>", unsafe_allow_html=True)
                st.markdown(f"</div>", unsafe_allow_html=True)
    
    with tab2:
        # ROM and time series plots
        st.pyplot(plot_rom_results(rom_results, angles_data))
        
        # Add heatmap
        st.subheader("Joint Angle Heatmap")
        st.pyplot(create_heatmap(angles_data))
    
    with tab3:
        # Raw data
        st.subheader("Raw Angle Data")
        st.dataframe(angles_data)
        
        # Download button for CSV
        csv = angles_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="angle_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    # Recommendations
    st.subheader("Insights & Recommendations")
    
    # Example insights (these would be more sophisticated in a real app)
    main_joints = ['Right knee', 'Left knee', 'Right hip', 'Left hip']
    insights = []
    
    for joint in main_joints:
        if joint in rom_results:
            rom = rom_results[joint]['rom']
            if rom < 90:
                insights.append(f"- **{joint}**: Limited range of motion ({rom:.1f}°). Consider mobility exercises.")
            elif rom > 150:
                insights.append(f"- **{joint}**: Excellent range of motion ({rom:.1f}°).")
            else:
                insights.append(f"- **{joint}**: Normal range of motion ({rom:.1f}°).")
    
    # Check asymmetry
    if 'Right knee' in rom_results and 'Left knee' in rom_results:
        r_knee = rom_results['Right knee']['rom']
        l_knee = rom_results['Left knee']['rom']
        asymmetry = abs(r_knee - l_knee)
        if asymmetry > 15:
            insights.append(f"- **Knee Asymmetry**: Significant difference between right and left knee ROM ({asymmetry:.1f}°). Consider targeted exercises for the more limited side.")
    
    if 'Right hip' in rom_results and 'Left hip' in rom_results:
        r_hip = rom_results['Right hip']['rom']
        l_hip = rom_results['Left hip']['rom']
        asymmetry = abs(r_hip - l_hip)
        if asymmetry > 15:
            insights.append(f"- **Hip Asymmetry**: Significant difference between right and left hip ROM ({asymmetry:.1f}°). Consider targeted exercises for the more limited side.")
    
    # Display insights
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.write("No specific insights available for this analysis.")

if __name__ == "__main__":
    main()