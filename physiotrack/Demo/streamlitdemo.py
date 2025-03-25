#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit web application for PhysioTrack.
Analyze range of motion and joint angles from webcam or uploaded video.
"""

import os
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import time
import threading
import queue
import subprocess
from datetime import datetime

# Import PhysioTrack
from physiotrack import PhysioTrack

# Global variables for webcam processing
webcam_running = False
webcam_process = None
webcam_results_path = None
webcam_frame_queue = queue.Queue(maxsize=10)

def create_config(video_path, result_dir, options, is_webcam=False):
    """Create a configuration dictionary for PhysioTrack."""
    return {
        'project': {
            'video_input': ['webcam'] if is_webcam else [str(video_path)],
            'px_to_m_person_height': options['height'],
            'visible_side': [options['visible_side']],
            'time_range': options['time_range'] if options['time_range'] else [],
            'webcam_id': options.get('webcam_id', 0),
            'input_size': [1280, 720],
            'load_trc_px': '',
            'compare': False
        },
        'process': {
            'multiperson': options['multiperson'],
            'show_realtime_results': options.get('show_realtime', True) if is_webcam else False,
            'save_vid': True,
            'save_img': False,
            'save_pose': True,
            'calculate_angles': True,
            'save_angles': True,
            'result_dir': str(result_dir)
        },
        'pose': {
            'pose_model': options['pose_model'],
            'mode': options['mode'],
            'det_frequency': options['det_frequency'],
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
            'joint_angles': options['joint_angles'],
            'segment_angles': options['segment_angles'],
            'display_angle_values_on': ['body', 'list'],
            'fontSize': 0.3,
            'flip_left_right': True,
            'correct_segment_angles_with_floor_angle': True
        },
        'px_to_meters_conversion': {
            'to_meters': options['to_meters'],
            'make_c3d': False,
            'calib_file': '',
            'floor_angle': 'auto',
            'xy_origin': ['auto'],
            'save_calib': True
        },
        'post-processing': {
            'interpolate': True,
            'filter': True,
            'show_graphs': False,
            'filter_type': options['filter_type'],
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
            'default_height': options['height'],
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

def analyze_rom(mot_file):
    """
    Analyze range of motion from a motion file.
    
    Args:
        mot_file: Path to the motion file
        
    Returns:
        rom_results: Dictionary of ROM results
        angles_data: DataFrame of angle data
    """
    # Skip the header lines
    with open(mot_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                header_rows = i
                break
    
    # Read the data
    angles_data = pd.read_csv(mot_file, sep='\t', skiprows=header_rows)
    
    # Calculate ROM for each joint
    rom_results = {}
    for col in angles_data.columns:
        if col == 'time':
            continue
        
        # Calculate min, max, and ROM
        min_val = angles_data[col].min()
        max_val = angles_data[col].max()
        rom = max_val - min_val
        
        rom_results[col] = {
            'min': min_val,
            'max': max_val,
            'rom': rom
        }
    
    return rom_results, angles_data

def display_video(video_path):
    """Display a video in Streamlit."""
    # Display video using HTML
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

def plot_rom_results(rom_results, angles_data):
    """Plot ROM results."""
    fig = plt.figure(figsize=(12, 10))
    
    # ROM bar chart
    plt.subplot(2, 1, 1)
    plt.bar(rom_results.keys(), [data['rom'] for data in rom_results.values()])
    plt.title('Range of Motion Analysis')
    plt.ylabel('ROM (degrees)')
    plt.xticks(rotation=45, ha='right')
    
    # Time series plot for key joints
    plt.subplot(2, 1, 2)
    for col in angles_data.columns:
        if col != 'time' and col in ['Right knee', 'Left knee', 'Right hip', 'Left hip', 'Trunk']:
            plt.plot(angles_data['time'], angles_data[col], label=col)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angle Time Series')
    plt.legend()
    plt.tight_layout()
    
    return fig

def plot_angle_curves(angles_data):
    """Plot angle curves for each joint."""
    # Create a figure with multiple subplots - one for each angle
    angle_cols = [col for col in angles_data.columns if col != 'time']
    
    # Determine the grid size
    n_angles = len(angle_cols)
    n_cols = 2
    n_rows = (n_angles + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, n_rows * 3))
    
    for i, angle in enumerate(angle_cols):
        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.plot(angles_data['time'], angles_data[angle])
        ax.set_title(f'{angle} Angle')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (°)')
        ax.grid(True)
        
        # Add min, max, and ROM text
        min_val = angles_data[angle].min()
        max_val = angles_data[angle].max()
        rom = max_val - min_val
        
        text_str = f'Min: {min_val:.1f}°\nMax: {max_val:.1f}°\nROM: {rom:.1f}°'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def start_webcam_processing(options, result_dir):
    """Start PhysioTrack processing on webcam in a separate process."""
    global webcam_process, webcam_running, webcam_results_path
    
    # Set up the configuration for webcam
    webcam_options = options.copy()
    webcam_options['show_realtime'] = True
    
    # Update results path to include timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    webcam_results_path = result_dir / f"webcam_session_{timestamp}"
    webcam_results_path.mkdir(exist_ok=True, parents=True)
    
    config = create_config(None, webcam_results_path, webcam_options, is_webcam=True)
    
    # Start the webcam processing as a subprocess
    # Serialize the config to a temporary file
    temp_config_file = result_dir / f"temp_config_{timestamp}.py"
    with open(temp_config_file, 'w') as f:
        f.write(f"import sys\nfrom physiotrack import PhysioTrack\n\nconfig = {config}\nPhysioTrack.process(config)")
    
    # Start subprocess
    webcam_process = subprocess.Popen([sys.executable, str(temp_config_file)])
    webcam_running = True
    
    return webcam_results_path

def stop_webcam_processing():
    """Stop the webcam processing."""
    global webcam_process, webcam_running
    
    if webcam_process and webcam_running:
        webcam_process.terminate()
        webcam_process = None
        webcam_running = False

def webcam_page():
    """Webcam live analysis page."""
    global webcam_running, webcam_results_path
    
    st.header("Webcam Live Analysis")
    
    # Options for webcam analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pose_model = st.selectbox(
            "Pose Model (Webcam)",
            ["body_with_feet", "whole_body", "body"],
            index=0,
            key="webcam_pose_model"
        )
        
        mode = st.selectbox(
            "Processing Mode (Webcam)",
            ["lightweight", "balanced", "performance"],
            index=0,  # Use lightweight for webcam
            key="webcam_mode"
        )
    
    with col2:
        webcam_id = st.number_input("Webcam ID", min_value=0, max_value=10, value=0, step=1)
        
        multiperson = st.checkbox("Multi-person Tracking (Webcam)", value=False, key="webcam_multiperson")
        
        to_meters = st.checkbox("Convert to Meters (Webcam)", value=True, key="webcam_to_meters")
    
    with col3:
        height = st.slider(
            "Subject Height (m) (Webcam)",
            min_value=1.0,
            max_value=2.2,
            value=1.7,
            step=0.05,
            key="webcam_height"
        )
        
        visible_side = st.selectbox(
            "Visible Side (Webcam)",
            ["auto", "front", "back", "left", "right", "none"],
            index=0,
            key="webcam_visible_side"
        )
        
        det_frequency = st.slider(
            "Detection Frequency (Webcam)",
            min_value=1,
            max_value=10,
            value=4,
            key="webcam_det_freq"
        )
    
    # Pre-select some angles for webcam for performance
    joint_angles = ['Right knee', 'Left knee', 'Right hip', 'Left hip']
    segment_angles = ['Trunk']
    
    # Set up webcam options
    filter_type = "butterworth"
    
    webcam_options = {
        'pose_model': pose_model,
        'mode': mode,
        'multiperson': multiperson,
        'to_meters': to_meters,
        'height': height,
        'visible_side': visible_side,
        'det_frequency': det_frequency,
        'filter_type': filter_type,
        'joint_angles': joint_angles,
        'segment_angles': segment_angles,
        'time_range': [],
        'webcam_id': webcam_id
    }
    
    # Create temp directory for results
    temp_dir = Path(tempfile.gettempdir()) / "physiotrack_webcam"
    temp_dir.mkdir(exist_ok=True)
    
    # Create placeholders for webcam feed and results
    webcam_status = st.empty()
    col1, col2 = st.columns(2)
    
    with col1:
        webcam_feed = st.empty()
    
    with col2:
        webcam_results = st.empty()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if not webcam_running:
            if st.button("Start Webcam Analysis"):
                webcam_status.info("Starting webcam analysis...")
                webcam_results_path = start_webcam_processing(webcam_options, temp_dir)
                webcam_status.success("Webcam analysis running. Look at the external window for real-time visualization.")
        else:
            if st.button("Stop Webcam Analysis"):
                stop_webcam_processing()
                webcam_status.info("Webcam analysis stopped.")
    
    with col2:
        if webcam_running or webcam_results_path:
            if st.button("Show Analysis Results"):
                webcam_status.info("Retrieving analysis results...")
                
                # Try to find the most recent results
                if webcam_results_path and webcam_results_path.exists():
                    # Find processed video
                    processed_videos = list(webcam_results_path.glob("*.mp4"))
                    if processed_videos:
                        latest_video = max(processed_videos, key=os.path.getmtime)
                        webcam_feed.video(str(latest_video))
                    
                    # Find and analyze angle files
                    mot_files = list(webcam_results_path.glob("*_angles_person*.mot"))
                    if mot_files:
                        latest_mot = max(mot_files, key=os.path.getmtime)
                        rom_results, angles_data = analyze_rom(latest_mot)
                        
                        # Display ROM results
                        results_df = pd.DataFrame([
                            {"Angle": angle, "Min (°)": data["min"], "Max (°)": data["max"], "ROM (°)": data["rom"]}
                            for angle, data in rom_results.items()
                        ])
                        
                        webcam_results.dataframe(results_df, use_container_width=True)
                        
                        # Create plot of results
                        fig = plot_rom_results(rom_results, angles_data)
                        st.pyplot(fig)
                    else:
                        webcam_results.warning("No angle data found yet. Try again in a few moments.")
                else:
                    webcam_status.error("No webcam results directory found.")
    
    # Information about webcam usage
    st.markdown("""
    ### Instructions:
    1. Click "Start Webcam Analysis" to begin live capture and analysis
    2. A separate window will open showing the real-time analysis
    3. Move through your range of motion exercises
    4. Click "Stop Webcam Analysis" when done
    5. Click "Show Analysis Results" to see your ROM data
    """)

def uploaded_video_page():
    """Uploaded video analysis page."""
    st.header("Video Upload Analysis")
    
    # Sidebar with options
    st.sidebar.header("Analysis Options")
    
    pose_model = st.sidebar.selectbox(
        "Pose Model",
        ["body_with_feet", "whole_body", "body"],
        index=0
    )
    
    mode = st.sidebar.selectbox(
        "Processing Mode",
        ["lightweight", "balanced", "performance"],
        index=2
    )
    
    multiperson = st.sidebar.checkbox("Multi-person Tracking", value=False)
    to_meters = st.sidebar.checkbox("Convert to Meters", value=True)
    
    height = st.sidebar.slider(
        "Subject Height (m)",
        min_value=1.0,
        max_value=2.2,
        value=1.7,
        step=0.05
    )
    
    visible_side = st.sidebar.selectbox(
        "Visible Side",
        ["auto", "front", "back", "left", "right", "none"],
        index=0
    )
    
    det_frequency = st.sidebar.slider(
        "Detection Frequency",
        min_value=1,
        max_value=10,
        value=4
    )
    
    filter_type = st.sidebar.selectbox(
        "Filter Type",
        ["butterworth", "gaussian", "loess", "median"],
        index=0
    )
    
    st.sidebar.header("Joint Angle Selection")
    
    all_joint_angles = ['Right ankle', 'Left ankle', 'Right knee', 'Left knee', 
                        'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder', 
                        'Right elbow', 'Left elbow', 'Right wrist', 'Left wrist']
    
    all_segment_angles = ['Right foot', 'Left foot', 'Right shank', 'Left shank', 
                          'Right thigh', 'Left thigh', 'Pelvis', 'Trunk', 
                          'Shoulders', 'Head', 'Right arm', 'Left arm', 
                          'Right forearm', 'Left forearm']
    
    joint_angles = st.sidebar.multiselect(
        "Joint Angles to Analyze",
        all_joint_angles,
        default=['Right knee', 'Left knee', 'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder']
    )
    
    segment_angles = st.sidebar.multiselect(
        "Segment Angles to Analyze",
        all_segment_angles,
        default=['Right thigh', 'Left thigh', 'Trunk']
    )
    
    time_range = st.sidebar.text_input(
        "Time Range (start end)",
        ""
    )
    if time_range:
        try:
            time_range = [float(x) for x in time_range.split()]
            if len(time_range) != 2:
                st.sidebar.error("Time range must be two values: start end")
                time_range = []
        except:
            st.sidebar.error("Invalid time range format. Use: start_time end_time")
            time_range = []
    else:
        time_range = []
    
    # Options dictionary
    options = {
        'pose_model': pose_model,
        'mode': mode,
        'multiperson': multiperson,
        'to_meters': to_meters,
        'height': height,
        'visible_side': visible_side,
        'det_frequency': det_frequency,
        'filter_type': filter_type,
        'joint_angles': joint_angles,
        'segment_angles': segment_angles,
        'time_range': time_range
    }
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Create a temporary directory to work with
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)
        
        # Save the uploaded video to the temporary directory
        temp_video_path = temp_path / uploaded_file.name
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Video uploaded successfully: {uploaded_file.name}")
        
        # Create results directory
        results_dir = temp_path / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Analyze button
        if st.button("Run Analysis"):
            with st.spinner("Processing video..."):
                try:
                    # Create configuration dictionary
                    config = create_config(temp_video_path, results_dir, options)
                    
                    # Process video with PhysioTrack
                    PhysioTrack.process(config)
                    
                    st.success("Processing complete!")
                    
                    # Find the processed video
                    processed_videos = list(results_dir.glob("*.mp4"))
                    if processed_videos:
                        processed_video = processed_videos[0]
                        st.subheader("Processed Video")
                        display_video(processed_video)
                    else:
                        st.error("No processed video found.")
                    
                    # Find and analyze angle files
                    mot_files = list(results_dir.glob("*_angles_person*.mot"))
                    if mot_files:
                        # Tabs for different persons if multiple are detected
                        if len(mot_files) > 1:
                            tabs = st.tabs([f"Person {i}" for i in range(len(mot_files))])
                            
                            for i, (tab, mot_file) in enumerate(zip(tabs, mot_files)):
                                with tab:
                                    rom_results, angles_data = analyze_rom(mot_file)
                                    
                                    # Display ROM results
                                    st.subheader(f"Range of Motion Results - Person {i}")
                                    results_df = pd.DataFrame([
                                        {"Angle": angle, "Min (°)": data["min"], "Max (°)": data["max"], "ROM (°)": data["rom"]}
                                        for angle, data in rom_results.items()
                                    ])
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Plot ROM
                                    st.subheader("Range of Motion Visualization")
                                    rom_fig = plot_rom_results(rom_results, angles_data)
                                    st.pyplot(rom_fig)
                                    
                                    # Plot individual angle curves
                                    st.subheader("Detailed Angle Curves")
                                    angles_fig = plot_angle_curves(angles_data)
                                    st.pyplot(angles_fig)
                        else:
                            # Single person case
                            mot_file = mot_files[0]
                            rom_results, angles_data = analyze_rom(mot_file)
                            
                            # Display ROM results
                            st.subheader("Range of Motion Results")
                            results_df = pd.DataFrame([
                                {"Angle": angle, "Min (°)": data["min"], "Max (°)": data["max"], "ROM (°)": data["rom"]}
                                for angle, data in rom_results.items()
                            ])
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Plot ROM
                            st.subheader("Range of Motion Visualization")
                            rom_fig = plot_rom_results(rom_results, angles_data)
                            st.pyplot(rom_fig)
                            
                            # Plot individual angle curves
                            st.subheader("Detailed Angle Curves")
                            angles_fig = plot_angle_curves(angles_data)
                            st.pyplot(angles_fig)
                    else:
                        st.error("No angle files found. Analysis may have failed.")
                
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Example video section
    st.header("No video to upload?")
    if st.button("Run Demo with Sample Video"):
        # Get the demo video path
        demo_dir = Path(__file__).resolve().parent
        demo_video = demo_dir / "Demo" / "lb-flexion.mp4"
        
        if not demo_video.exists():
            st.error(f"Demo video not found at {demo_video}")
        else:
            with st.spinner("Processing demo video..."):
                try:
                    # Create results directory
                    results_dir = demo_dir / "results"
                    results_dir.mkdir(exist_ok=True)
                    
                    # Create configuration dictionary
                    config = create_config(demo_video, results_dir, options)
                    
                    # Process video with PhysioTrack
                    PhysioTrack.process(config)
                    
                    st.success("Demo processing complete!")
                    
                    # Find the processed video
                    processed_videos = list(results_dir.glob("*.mp4"))
                    if processed_videos:
                        processed_video = processed_videos[0]
                        st.subheader("Processed Demo Video")
                        display_video(processed_video)
                    
                    # Find and analyze angle files
                    mot_files = list(results_dir.glob("*_angles_person*.mot"))
                    if mot_files:
                        mot_file = mot_files[0]
                        rom_results, angles_data = analyze_rom(mot_file)
                        
                        # Display ROM results
                        st.subheader("Demo Range of Motion Results")
                        results_df = pd.DataFrame([
                            {"Angle": angle, "Min (°)": data["min"], "Max (°)": data["max"], "ROM (°)": data["rom"]}
                            for angle, data in rom_results.items()
                        ])
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Plot ROM
                        st.subheader("Demo Range of Motion Visualization")
                        rom_fig = plot_rom_results(rom_results, angles_data)
                        st.pyplot(rom_fig)
                        
                        # Plot individual angle curves
                        st.subheader("Demo Detailed Angle Curves")
                        angles_fig = plot_angle_curves(angles_data)
                        st.pyplot(angles_fig)
                    else:
                        st.error("No angle files found for demo. Analysis may have failed.")
                
                except Exception as e:
                    st.error(f"Error during demo processing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

def main():
    st.set_page_config(
        page_title="PhysioTrack - Range of Motion Analysis",
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("PhysioTrack: Range of Motion Analysis")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Webcam Live Analysis", "Video Upload Analysis"])
    
    with tab1:
        webcam_page()
    
    with tab2:
        uploaded_video_page()
    
    # Footer
    st.markdown("---")
    st.markdown("PhysioTrack - Range of Motion Analysis Tool")
    st.markdown("Made with ❤️ for physiotherapy assessment")

    # Handle clean up when app stops
    def cleanup():
        stop_webcam_processing()
    
    # Register cleanup function
    import atexit
    atexit.register(cleanup)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # Make sure to clean up webcam process if the app crashes
        stop_webcam_processing()