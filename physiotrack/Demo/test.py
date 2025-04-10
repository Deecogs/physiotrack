#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PhysioTrack Webcam Inference Demo
---------------------------------
This script captures video from your webcam, runs real-time pose estimation and angle 
calculation using PhysioTrack, and displays the results live on screen.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Import PhysioTrack
from physiotrack import PhysioTrack

def main():
    """
    Run PhysioTrack with webcam input for real-time pose and ROM analysis.
    """
    print("Starting PhysioTrack webcam inference...")
    
    # Create configuration dictionary
    config = {
        'project': {
            'video_input': ['webcam'],
            'px_to_m_person_height': 1.7,
            'visible_side': ['auto'],
            'time_range': [],
            'webcam_id': 0,  # Change this if you have multiple webcams
            'input_size': [1280, 720],  # Adjust based on your webcam capabilities
            'load_trc_px': '',
            'compare': False
        },
        'process': {
            'multiperson': False,  # Set to True to track multiple people
            'show_realtime_results': True,  # Must be True to see video output
            'save_vid': False,  # Set to True if you want to save the video
            'save_img': False,
            'save_pose': False,
            'calculate_angles': True,
            'save_angles': False,
            'result_dir': str(Path.cwd() / 'physiotrack_output')
        },
        'pose': {
            'pose_model': 'body_with_feet',  # Use body with feet for better ROM analysis
            'mode': 'lightweight',  # Can be 'lightweight', 'balanced', or 'performance'
            'det_frequency': 4,  # Detect persons every 4 frames
            'device': 'auto',  # 'auto', 'cpu', 'cuda', etc.
            'backend': 'auto',  # 'auto', 'openvino', 'onnxruntime', etc.
            'tracking_mode': 'physiotrack',
            'deepsort_params': """{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, 'max_cosine_distance':0.3, 'nn_budget':200, 'max_iou_distance':0.8, 'embedder_gpu': True}""",
            'keypoint_likelihood_threshold': 0.3,
            'average_likelihood_threshold': 0.5,
            'keypoint_number_threshold': 0.3,
            'slowmo_factor': 1
        },
        'angles': {
            'joint_angles': [
                'Right knee', 'Left knee', 'Right hip', 'Left hip', 
                'Right shoulder', 'Left shoulder', 'Right elbow', 'Left elbow'
            ],
            'segment_angles': [
                'Right thigh', 'Left thigh', 'Trunk'
            ],
            'display_angle_values_on': ['body', 'list'],
            'fontSize': 0.4,
            'flip_left_right': True,
            'correct_segment_angles_with_floor_angle': True
        },
        'px_to_meters_conversion': {
            'to_meters': False,  # Set to True if you want to convert to meters
            'make_c3d': False,
            'calib_file': '',
            'floor_angle': 'auto',
            'xy_origin': ['auto'],
            'save_calib': False
        },
        'post-processing': {
            'interpolate': True,
            'filter': False,  # Filtering might slow down real-time performance
            'show_graphs': False,
            'filter_type': 'butterworth',
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
            'use_contacts_muscles': False,
            'participant_mass': [70.0],
            'right_left_symmetry': True,
            'default_height': 1.70,
            'fastest_frames_to_remove_percent': 0.1,
            'close_to_zero_speed_px': 50,
            'close_to_zero_speed_m': 0.2,
            'large_hip_knee_angles': 45,
            'trimmed_extrema_percent': 0.5,
            'osim_setup_path': '../OpenSim_setup'
        },
        'logging': {
            'use_custom_logging': False
        }
    }
    
    # Create the output directory if it doesn't exist
    Path(config['process']['result_dir']).mkdir(parents=True, exist_ok=True)
    
    print("Press 'q' to quit the application")
    print("Starting webcam capture and pose estimation...")
    
    try:
        # Process video with PhysioTrack
        PhysioTrack.process(config)
        print("Processing complete.")
    except Exception as e:
        print(f"Error running PhysioTrack: {e}")
    
if __name__ == "__main__":
    main()