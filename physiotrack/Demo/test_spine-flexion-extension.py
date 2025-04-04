#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the Range of Motion (ROM) analysis functionality of PhysioTrack.
This is a demo that shows how to load a video file, process it, and analyze the ROM.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Import PhysioTrack
from physiotrack import PhysioTrack

# mode = 'performance' # 'lightweight', 'balanced', 'performance'
def main():
    """Run ROM analysis on a demo video."""
    print("Starting ROM analysis on demo video...")
    
    # Get the demo video path
    demo_dir = Path(__file__).resolve().parent
    print(f"demo_dir {demo_dir}")
    demo_video = demo_dir / "Spine-flexion-extension-side.mp4"
    # "demo.mp4"
    # /Users/chandansharma/Desktop/workspace/metashape/projects/dc-pose/dochq/physiotrack/physiotrack/Demo/lb-flexion.mp4
    if not demo_video.exists():
        print(f"Error: Demo video not found at {demo_video}")
        return
    
    # Create a configuration dictionary for PhysioTrack
    config_dict = {
        'project': {
            'video_input': [str(demo_video)],
            'px_to_m_person_height': 1.65,
            'visible_side': ['auto'],
            'time_range': [],  # Analyze the whole video
            'video_dir': str(demo_dir),  # Make sure this is a string
            'webcam_id': 0,
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
            'result_dir': str(demo_dir / 'results/demo_PhysioTrack')
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
            'use_contacts_muscles': True,
            'participant_mass': [67.0, 55.0],
            'right_left_symmetry': True,
            'default_height': 1.70,
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
            'use_custom_logging': False
        }
    }
    
    # Run PhysioTrack with the configuration
    PhysioTrack.process(config_dict)
    
    print("Processing complete. Now analyzing ROM...")
    
    # Find the angles.mot file
    results_dir = Path(config_dict['process']['result_dir'])
    print("results_dir", results_dir)

    # # Name the subfolder as the video file name_PhysioTrack
    subfolder = "Spine-flexion-extension-side_PhysioTrack"  # Ensure we look in the correct subfolder
    full_results_dir = results_dir / subfolder
    print("Looking for results in:", full_results_dir)

    mot_files = list(full_results_dir.glob("*_angles*.mot"))
    
    if not mot_files:
        print("No angle files found. Processing may have failed.")
        return
    
    # Load the angles file
    mot_file = mot_files[0]
    print(f"Analyzing ROM from {mot_file}")
    
    # Skip the header lines
    with open(mot_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                header_rows = i
                break
    
    # Read the data
    angles_data = pd.read_csv(mot_file, sep='\t', skiprows=header_rows)

    # Ensure time column is float
    angles_data['time'] = angles_data['time'].astype(float)

    window_size = 0.4 # seconds
    time_step = angles_data['time'].iloc[1] - angles_data['time'].iloc[0]

    # Calculate window size in terms of indices
    window_size_idx = int(window_size / time_step)
    
    # Calculate ROM for each joint
    rom_results = {}

    # Define the output file path
    output_file = full_results_dir / "rom_results.json"
    log_file = full_results_dir / "logs.json"

    logs = {"ROM Calculation Logs": []}

    for col in angles_data.columns:
        if col == 'time':
            continue

        # Apply transformation (180 - angle)
        angles_data[f'{col}_transformed'] = 180 - angles_data[col]

        # Compute rolling mean over the given window size
        angles_data[f'{col}_avg'] = angles_data[f'{col}_transformed'].rolling(window=window_size_idx, min_periods=1).mean()
        # Find min and max from the averaged values
        min_val = angles_data[f'{col}_avg'].min()
        max_val = angles_data[f'{col}_avg'].max()
        rom = max_val - min_val

        rom_results[col] = {
            'min': min_val,
            'max': max_val,
            'rom': rom
        }

        output_string = f"ROM for {col}: {rom:.2f} degrees (Min: {min_val:.2f}, Max: {max_val:.2f})"
        print(output_string)

        logs["ROM Calculation Logs"].append({
            "joint": col,
            "min_value": min_val,
            "max_value": max_val,
            "rom": rom
        })

    # Save results to JSON files
    with open(output_file, 'w') as f:
        json.dump(rom_results, f, indent=4)

    with open(log_file, 'w') as log:
        json.dump(logs, log, indent=4)

    print(f"ROM results saved to {output_file}")
    print(f"Detailed logs saved to {log_file}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # ROM bar chart
    plt.subplot(2, 1, 1)
    plt.bar(rom_results.keys(), [data['rom'] for data in rom_results.values()])
    plt.title('Range of Motion Analysis')
    plt.ylabel('ROM (degrees)')
    plt.xticks(rotation=45, ha='right')
    
    # Time series plot for key joints
    plt.subplot(2, 1, 2)
    for col in ['Right knee', 'Left knee', 'Right hip', 'Left hip']:
        if col in angles_data.columns:
            plt.plot(angles_data['time'], angles_data[col], label=col)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angle Time Series')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(full_results_dir / 'rom_analysis.png')
    print(f"ROM analysis plot saved to {full_results_dir / 'rom_analysis.png'}")
    # plt.savefig(results_dir / 'rom_analysis.png')
    # print(f"ROM analysis plot saved to {results_dir / 'rom_analysis.png'}")
    
    plt.show()
    
if __name__ == "__main__":
    main()