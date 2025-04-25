#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script showing how to use PhysioTrack with webcam input.
This script uses PhysioTrack's process function to handle webcam input and tracking.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Import PhysioTrack
from physiotrack.PhysioTrack import process

def main():
    """Run PhysioTrack analysis on webcam input."""
    print("Starting PhysioTrack webcam analysis...")
    
    # Get the current directory for results
    current_dir = Path(__file__).resolve().parent
    result_dir = str(current_dir / 'results/webcam_PhysioTrack')
    
    # Create a configuration dictionary for PhysioTrack
    config_dict = {
        'logging': {
            'use_custom_logging': False
        },
        'project': {
            'video_input': ['webcam'],  # Specify webcam input
            'px_to_m_person_height': 1.70,
            'visible_side': ['auto'],
            'time_range': [],  # Analyze continuously
            'video_dir': str(current_dir),
            'webcam_id': 0,  # Use default webcam
            'input_size': [1280, 720],
            'load_trc_px': '',
            'compare': False
        },
        'process': {
            'multiperson': False,  # Focus on one person
            'show_realtime_results': True,
            'save_vid': True,
            'save_img': False,
            'save_pose': True,
            'calculate_angles': True,
            'save_angles': True,
            'result_dir': str(result_dir)
        },
        'pose': {
            'pose_model': 'body_with_feet',
            'mode': 'performance',
            'det_frequency': 4,
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
            'joint_angles': ['Right knee', 'Left knee', 'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder'],
            'segment_angles': ['Right thigh', 'Left thigh', 'Trunk'],
            'display_angle_values_on': ['body', 'list'],
            'fontSize': 0.3,
            'flip_left_right': True,
            'correct_segment_angles_with_floor_angle': True
        },
        'px_to_meters_conversion': {
            'to_meters': True,
            'make_c3d': True,
            'calib_file': '',
            'floor_angle': 'auto',
            'xy_origin': ['auto'],
            'save_calib': True
        },
        'post-processing': {
            'interpolate': True,
            'filter': True,
            'show_graphs': True,
            'filter_type': 'butterworth',
            'butterworth': {'order': 4, 'cut_off_frequency': 6},
            'gaussian': {'sigma_kernel': 1},
            'loess': {'nb_values_used': 5},
            'median': {'kernel_size': 3},
            'interp_gap_smaller_than': 10,
            'fill_large_gaps_with': 'last_value'
        }
    }

    # Create Streamlit UI
    st.title("PhysioTrack Webcam Analysis")
    
    # Add webcam configuration in sidebar
    st.sidebar.header("Webcam Configuration")
    config_dict['project']['webcam_id'] = st.sidebar.number_input("Webcam ID", min_value=0, value=0)
    
    # Add a button to start processing
    if st.button("Start Webcam Analysis"):
        try:
            # Call PhysioTrack's process function with webcam configuration
            process(config_dict)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
