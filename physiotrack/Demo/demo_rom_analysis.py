#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script showing how to use the ROM data functionality in PhysioTrack.
This script shows how to process a video, generate ROM data, and analyze the results.
The ROM data is saved for each frame/time point in a nested JSON structure.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import PhysioTrack
from physiotrack import PhysioTrack
from physiotrack.process import read_rom_data

def main():
    """Run ROM analysis on a demo video."""
    print("Starting ROM analysis on demo video...")
    
    # Get the demo video path
    demo_dir = Path(__file__).resolve().parent
    print(f"Demo directory: {demo_dir}")
    demo_video = demo_dir / "demo-spine-flexion-1.mp4"
    
    if not demo_video.exists():
        print(f"Error: Demo video not found at {demo_video}")
        print("Please place a demo video file named 'demo-spine-flexion-1.mp4' in the Demo directory.")
        return
    
    # Create a configuration dictionary for PhysioTrack
    config_dict = {
        'project': {
            'video_input': [str(demo_video)],
            'px_to_m_person_height': 1.70,
            'visible_side': ['auto'],
            'time_range': [],  # Analyze the whole video
            'video_dir': str(demo_dir),
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
            'correct_segment_angles_with_floor_angle': True,
            # 'test_name': 'lower_back_flexion'
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
    
    print("Processing complete. Now analyzing ROM data...")
    
    # Find the ROM data file
    results_dir = Path(config_dict['process']['result_dir'])
    subfolder = f"demo-spine-flexion-1_PhysioTrack"
    full_results_dir = results_dir / subfolder
    
    rom_files = list(full_results_dir.glob("*_rom_data.json"))
    
    if not rom_files:
        print("No ROM data files found. Processing may have failed.")
        return
    
    # Load and analyze ROM data
    rom_file = rom_files[0]
    print(f"Analyzing ROM data from {rom_file}")
    
    # Read the ROM data
    rom_data = read_rom_data(str(rom_file))
    
    # Convert time-based data to dataframes for easier analysis
    times = [float(t) for t in rom_data.keys()]
    times.sort()  # Ensure times are in order
    
    # First time point to get joint names
    first_time = str(times[0])
    joint_names = list(rom_data[first_time]['angles'].keys())
    
    # Create a DataFrame with time and angle data
    df = {'time': times}
    
    # Extract angle data for each joint over time
    for joint in joint_names:
        df[joint] = []
        for t in times:
            time_str = str(t)
            if time_str in rom_data and joint in rom_data[time_str]['angles']:
                df[joint].append(rom_data[time_str]['angles'][joint])
            else:
                df[joint].append(np.nan)  # Handle missing data
    
    # Convert to DataFrame
    df = pd.DataFrame(df)
    
    # Calculate min, max, and range for each joint
    summary_data = {}
    for joint in joint_names:
        min_val = df[joint].min()
        max_val = df[joint].max()
        rom_range = max_val - min_val
        
        summary_data[joint] = {
            'min': min_val,
            'max': max_val,
            'rom': rom_range
        }
    
    # Display the ROM summary
    print("\nROM Analysis Summary:")
    print("---------------------")
    for joint, data in summary_data.items():
        print(f"\nJoint: {joint}")
        print(f"  Min: {data['min']:.1f}°")
        print(f"  Max: {data['max']:.1f}°")
        print(f"  ROM: {data['rom']:.1f}°")
    
    # Plot the angle data over time
    plt.figure(figsize=(12, 10))
    
    # ROM range bar chart
    plt.subplot(2, 1, 1)
    joints = list(summary_data.keys())
    rom_ranges = [data['rom'] for data in summary_data.values()]
    
    bars = plt.bar(joints, rom_ranges)
    plt.title('Range of Motion Analysis')
    plt.ylabel('ROM Range (degrees)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, rom_ranges):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{value:.1f}°', ha='center', va='bottom')
    
    # Plot angle values over time
    plt.subplot(2, 1, 2)
    
    # Plot each joint's angle over time
    for joint in joint_names:
        plt.plot(df['time'], df[joint], label=joint)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles Over Time')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    
    # Save plot
    output_plot = full_results_dir / "rom_analysis_plot.png"
    plt.savefig(output_plot)
    print(f"\nROM analysis plot saved to {output_plot}")
    
    # Save the time series data as CSV for further analysis
    csv_path = full_results_dir / "angle_time_series.csv"
    df.to_csv(csv_path, index=False)
    print(f"Angle time series data saved to {csv_path}")
    
    plt.show()
    
if __name__ == "__main__":
    main() 