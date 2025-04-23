## INIT
from pathlib import Path
import sys
import logging
import json
import ast
import shutil
import os
from functools import partial
from datetime import datetime
import itertools as it
from tqdm import tqdm
from collections import defaultdict
from anytree import RenderTree

import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Custom
from deep_sort_realtime.deepsort_tracker import DeepSort

from physiotrack.Utilities import filter
from physiotrack.Utilities.common import *
from physiotrack.Utilities.skeletons import *
from physiotrack.Utilities.common import angle_dict


def initialize_pose_tracker(config_dict):
    '''
    Initialize pose tracker and related model/config objects for use in web-based frame processing.
    Returns: pose_tracker, angle_dict, keypoints_ids, keypoints_names, model_root
    '''
    # (This is a minimal version, adapt as needed)
    pose_model = config_dict['pose'].get('pose_model', 'body_with_feet')
    mode = config_dict['pose'].get('mode', 'balanced')
    det_frequency = config_dict['pose'].get('det_frequency', 4)
    backend = config_dict['pose'].get('backend', 'auto')
    device = config_dict['pose'].get('device', 'auto')
    # Use angle_dict from physiotrack.Utilities.common

    # Map pose_model to both RTMLib model class and skeleton root Node
    from physiotrack.Utilities.skeletons import HALPE_26, COCO_133_WRIST, COCO_17
    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        ModelClass = BodyWithFeet
        model_root = HALPE_26
    elif pose_model.upper() == 'WHOLE_BODY_WRIST':
        ModelClass = Wholebody
        model_root = COCO_133_WRIST
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY'):
        ModelClass = Wholebody
        model_root = COCO_133_WRIST
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        ModelClass = Body
        model_root = COCO_17
    else:
        raise ValueError(f"Invalid model_type: {pose_model}")
    # Initialize pose tracker
    pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
    # Get keypoints info using the skeleton root node
    keypoints_ids = [node.id for _, _, node in RenderTree(model_root)]
    keypoints_names = [node.name for _, _, node in RenderTree(model_root)]
    return pose_tracker, angle_dict, keypoints_ids, keypoints_names, model_root


def generate_rom_data(angle_data, test_name, rom_path):
    '''
    Generate a ROM data json file from angle data, with data for each frame/time point.
    
    INPUTS:
    - angle_data: pd.DataFrame. The angles data with time column and angle columns
    - test_name: str. The name of the test (e.g., 'lower_back_flexion')
    - rom_path: str. The path where to save the ROM data json file
    
    OUTPUT:
    - rom_data: dict. The ROM data that has been written to the json file
    '''
    import json
    import numpy as np
    
    # Initialize the ROM data structure
    rom_data = {}
    
    # Calculate window size in seconds and convert to indices
    window_size = 0.4  # seconds
    if 'trunk' in angle_data.columns:
        time_step = angle_data['time'].iloc[1] - angle_data['time'].iloc[0]
        window_size_idx = int(window_size / time_step)
        
        # Apply transformation (180 - angle) as in test_rom_analysis.py
        transformed_angles = 180 - angle_data['trunk']
        
        # Compute rolling mean over the window
        smoothed_angles = transformed_angles.rolling(window=window_size_idx, min_periods=1).mean()
        
        # Calculate the running min/max values and ROM at each time point
        running_min = np.zeros(len(smoothed_angles))
        running_max = np.zeros(len(smoothed_angles))
        running_rom = np.zeros(len(smoothed_angles))
        
        for i in range(len(smoothed_angles)):
            # Consider all angles from start up to current point
            current_segment = smoothed_angles.iloc[0:i+1]
            
            if not current_segment.empty:
                # Calculate min, max and rom from start to current point
                current_min = current_segment.min()
                current_max = current_segment.max()
                current_rom = current_max - current_min
                
                running_min[i] = current_min
                running_max[i] = current_max
                running_rom[i] = current_rom
    
    # Process each time point/frame
    for idx, row in angle_data.iterrows():
        time_val = str(round(row['time'], 3))  # Use time as key, rounded to 3 decimal places
        
        # Initialize angles dict for this time point
        angles_dict = {}
        
        # Collect all angles for this time point
        for col in angle_data.columns:
            if col == 'time':
                continue
            
            # Add angle to the angles dictionary
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                angles_dict[col] = None
            else:
                angles_dict[col] = float(round(val, 1))
        
        # Default ROM values
        rom_min = 0.0
        rom_max = 0.0
        rom_range = 0.0
        
        # Set ROM values if trunk angle is available and we've calculated them
        if 'trunk' in angle_data.columns and idx < len(running_rom):
            rom_min = float(running_min[idx])
            rom_max = float(running_max[idx])
            rom_range = float(running_rom[idx])
            
        # Create entry for this time point with all angles together
        rom_data[time_val] = {
            "test": test_name,
            "is_ready": True,
            "angles": angles_dict,
            "ROM": [rom_min, rom_max],  # Update with calculated values for trunk
            "rom_range": rom_range,  # Update with calculated ROM for trunk
            "position_valid": True,
            "guidance": "Good posture",
            "posture_message": "Good posture",
            "ready_progress": 100,
            "status": "success"
        }
    
    # Log the final ROM calculation
    if 'trunk' in angle_data.columns and len(running_rom) > 0:
        final_rom = running_rom[-1]
        final_min = running_min[-1]
        final_max = running_max[-1]
        logging.info(f"ROM for trunk: {final_rom:.2f} degrees (Min: {final_min:.2f}, Max: {final_max:.2f})")
    
    # Write to JSON file
    with open(rom_path, 'w') as json_file:
        json.dump(rom_data, json_file, indent=4)
    
    logging.info(f'ROM data saved to {rom_path}.')
    return rom_data

def process_frame(frame, pose_tracker, config_dict, angle_dict, keypoints_ids, keypoints_names, model_root):
    '''
    Process a single frame: detect pose, compute angles, draw overlays.
    Args:
        frame (np.ndarray): Input image (BGR).
        pose_tracker: Initialized pose tracker object.
        config_dict (dict): Configuration dictionary.
        angle_dict (dict): Angle definitions.
        keypoints_ids (list): List of keypoint ids.
        keypoints_names (list): List of keypoint names.
    Returns:
        np.ndarray: Processed frame (with overlays).
    '''
    keypoint_likelihood_threshold = config_dict['pose'].get('keypoint_likelihood_threshold', 0.3)
    keypoint_number_threshold = config_dict['pose'].get('keypoint_number_threshold', 0.3)
    average_likelihood_threshold = config_dict['pose'].get('average_likelihood_threshold', 0.5)
    angle_names = config_dict['angles'].get('joint_angles', []) + config_dict['angles'].get('segment_angles', [])
    angle_names = [a.lower() for a in angle_names]
    display_angle_values_on = config_dict['angles'].get('display_angle_values_on', ['body', 'list'])
    fontSize = config_dict['angles'].get('fontSize', 0.3)
    thickness = 1 if fontSize < 0.8 else 2
    flip_left_right = config_dict['angles'].get('flip_left_right', True)
    calculate_angles = config_dict['process'].get('calculate_angles', True)

    # Detect pose
    keypoints, scores = pose_tracker(frame)
    valid_X, valid_Y, valid_scores = [], [], []
    valid_X_flipped, valid_angles = [], []
    for person_idx in range(len(keypoints)):
        # Remove low-confidence keypoints
        person_X, person_Y = np.where(scores[person_idx][:, np.newaxis] < keypoint_likelihood_threshold, np.nan, keypoints[person_idx]).T
        person_scores = np.where(scores[person_idx] < keypoint_likelihood_threshold, np.nan, scores[person_idx])
        enough_good_keypoints = len(person_scores[~np.isnan(person_scores)]) >= len(person_scores) * keypoint_number_threshold
        scores_of_good_keypoints = person_scores[~np.isnan(person_scores)]
        average_score_of_remaining_keypoints_is_enough = (np.nanmean(scores_of_good_keypoints) if len(scores_of_good_keypoints) > 0 else 0) >= average_likelihood_threshold
        if not enough_good_keypoints or not average_score_of_remaining_keypoints_is_enough:
            person_X = np.full_like(person_X, np.nan)
            person_Y = np.full_like(person_Y, np.nan)
            person_scores = np.full_like(person_scores, np.nan)
        # Compute angles
        if calculate_angles:
            if flip_left_right:
                try:
                    Ltoe_idx = keypoints_ids[keypoints_names.index('LBigToe')]
                    LHeel_idx = keypoints_ids[keypoints_names.index('LHeel')]
                    Rtoe_idx = keypoints_ids[keypoints_names.index('RBigToe')]
                    RHeel_idx = keypoints_ids[keypoints_names.index('RHeel')]
                    L_R_direction_idx = [Ltoe_idx, LHeel_idx, Rtoe_idx, RHeel_idx]
                    person_X_flipped = flip_left_right_direction(person_X, L_R_direction_idx, keypoints_names, keypoints_ids)
                except Exception:
                    person_X_flipped = person_X.copy()
            else:
                person_X_flipped = person_X.copy()
            person_angles = []
            new_keypoints_names, new_keypoints_ids = keypoints_names.copy(), keypoints_ids.copy()
            for kpt in ['Neck', 'Hip']:
                if kpt not in new_keypoints_names:
                    person_X_flipped, person_Y, person_scores = add_neck_hip_coords(kpt, person_X_flipped, person_Y, person_scores, new_keypoints_ids, new_keypoints_names)
                    person_X, _, _ = add_neck_hip_coords(kpt, person_X, person_Y, person_scores, new_keypoints_ids, new_keypoints_names)
                    new_keypoints_names.append(kpt)
                    new_keypoints_ids.append(len(person_X_flipped)-1)
            for ang_name in angle_names:
                ang_params = angle_dict.get(ang_name)
                kpts = ang_params[0]
                if not any(item not in new_keypoints_names for item in kpts):
                    ang = compute_angle(ang_name, person_X_flipped, person_Y, angle_dict, new_keypoints_ids, new_keypoints_names)
                else:
                    ang = np.nan
                person_angles.append(ang)
            valid_angles.append(person_angles)
            valid_X_flipped.append(person_X_flipped)
        valid_X.append(person_X)
        valid_Y.append(person_Y)
        valid_scores.append(person_scores)
    # Draw overlays
    img = frame.copy()
    img = draw_bounding_box(img, valid_X, valid_Y, fontSize=fontSize, thickness=thickness)
    img = draw_keypts(img, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
    img = draw_skel(img, valid_X, valid_Y, model_root)
    if calculate_angles:
        img = draw_angles(img, valid_X, valid_Y, valid_angles, valid_X_flipped, new_keypoints_ids, new_keypoints_names, angle_names, display_angle_values_on=display_angle_values_on, fontSize=fontSize, thickness=thickness)
    # Prepare angle dict for first person (or None)
    angle_dict_example = None
    if calculate_angles and valid_angles and len(valid_angles) > 0:
        angle_dict_example = {name: float(val) if not (val is None or (isinstance(val, float) and np.isnan(val))) else None for name, val in zip(angle_names, valid_angles[0])}
    return img, angle_dict_example

def setup_pose_tracker(ModelClass, det_frequency, mode, tracking, backend, device):
    '''
    Set up the RTMLib pose tracker with the appropriate model and backend.
    If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino

    INPUTS:
    - ModelClass: class. The RTMlib model class to use for pose detection (Body, BodyWithFeet, Wholebody)
    - det_frequency: int. The frequency of pose detection (every N frames)
    - mode: str. The mode of the pose tracker ('lightweight', 'balanced', 'performance')
    - tracking: bool. Whether to track persons across frames with RTMlib tracker
    - backend: str. The backend to use for pose detection (onnxruntime, openvino, opencv)
    - device: str. The device to use for pose detection (cpu, cuda, rocm, mps)

    OUTPUTS:
    - pose_tracker: PoseTracker. The initialized pose tracker object    
    '''

    backend, device = setup_backend_device(backend=backend, device=device)

    # Initialize the pose tracker with Halpe26 model
    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=False)
        
    return pose_tracker

def setup_backend_device(backend='auto', device='auto'):
    '''
    Set up the backend and device for the pose tracker based on the availability of hardware acceleration.

    If device and backend are not specified, they are automatically set up in the following order of priority:
    1. GPU with CUDA and ONNXRuntime backend (if CUDAExecutionProvider is available)
    2. GPU with ROCm and ONNXRuntime backend (if ROCMExecutionProvider is available, for AMD GPUs)
    3. GPU with MPS or CoreML and ONNXRuntime backend (for macOS systems)
    4. CPU with OpenVINO backend (default fallback)
    '''

    if device!='auto' and backend!='auto':
        device = device.lower()
        backend = backend.lower()

    if device=='auto' or backend=='auto':
        if device=='auto' and backend!='auto' or device!='auto' and backend=='auto':
            logging.warning(f"If you set device or backend to 'auto', you must set the other to 'auto' as well. Both device and backend will be determined automatically.")

        try:
            import torch
            import onnxruntime as ort
            if torch.cuda.is_available() == True and 'CUDAExecutionProvider' in ort.get_available_providers():
                device = 'cuda'
                backend = 'onnxruntime'
                logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
            elif torch.cuda.is_available() == True and 'ROCMExecutionProvider' in ort.get_available_providers():
                device = 'rocm'
                backend = 'onnxruntime'
                logging.info(f"\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
            else:
                raise ValueError("No CUDA or ROCm available")
        except:
            try:
                import onnxruntime as ort
                if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                    device = 'mps'
                    backend = 'onnxruntime'
                    logging.info(f"\nValid MPS installation found: using ONNXRuntime backend with GPU.")
                else:
                    raise ValueError("No MPS available")
            except:
                device = 'cpu'
                backend = 'openvino'
                logging.info(f"\nNo valid GPU acceleration found: using OpenVINO backend with CPU.")
        
    return backend, device

def compute_angle(ang_name, person_X_flipped, person_Y, angle_dict, keypoints_ids, keypoints_names):
    '''
    Compute the angles from the 2D coordinates of the keypoints.
    Takes into account which side the participant is facing.
    Takes into account the offset and scaling of the angle from angle_dict.
    Requires points_to_angles function (see common.py)

    INPUTS:
    - ang_name: str. The name of the angle to compute
    - person_X_flipped: list of x coordinates after flipping if needed
    - person_Y: list of y coordinates
    - angle_dict: dict. The dictionary of angles to compute (name: [keypoints, type, offset, scaling])
    - keypoints_ids: list of keypoint ids (see skeletons.py)
    - keypoints_names: list of keypoint names (see skeletons.py)

    OUTPUT:
    - ang: float. The computed angle
    '''

    ang_params = angle_dict.get(ang_name)
    if ang_params is not None:
        try:
            if ang_name in ['pelvis', 'trunk', 'shoulders']:
                angle_coords = [[np.abs(person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]]), person_Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0]]
            else:
                angle_coords = [[person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]], person_Y[keypoints_ids[keypoints_names.index(kpt)]]] for kpt in ang_params[0]]
            ang = fixed_angles(angle_coords, ang_name)
        except:
            ang = np.nan    
    else:
        ang = np.nan
    
    return ang

def draw_angles(img, valid_X, valid_Y, valid_angles, valid_X_flipped, keypoints_ids, keypoints_names, angle_names, display_angle_values_on=['body', 'list'], colors=[(255, 0, 0)], fontSize=0.3, thickness=1, rom_data=None):
    '''
    Draw only the relevant angles on the image.
    Angles: 
    - Right Hip
    - Left Hip
    - Right Knee
    - Left Knee

    INPUTS:
    - img: opencv image
    - valid_X: list of list of x coordinates
    - valid_Y: list of list of y coordinates
    - valid_angles: list of list of angles
    - valid_X_flipped: list of list of x coordinates after flipping if needed
    - keypoints_ids: list of keypoint ids (see skeletons.py)
    - keypoints_names: list of keypoint names (see skeletons.py)
    - angle_names: list of angle names
    - display_angle_values_on: list of str. 'body' and/or 'list' (display angles on the body)
    - colors: list of colors to cycle through

    OUTPUT:
    - img: image with angles
    '''

     # Define the relevant angles based on the console output
    relevant_angle_names = {
        "right knee",
        "left knee",
        "right hip",
        "left hip"
    }

    color_cycle = it.cycle(colors)
    for person_id, (X, Y, angles, X_flipped) in enumerate(zip(valid_X, valid_Y, valid_angles, valid_X_flipped)):
        c = next(color_cycle)

        if not np.isnan(X).all():
            # person label
            # if 'list' in display_angle_values_on:
            #     person_label_position = (int(10 + fontSize*150/0.3*person_id), int(fontSize*50))
            #     cv2.putText(img, f'person {person_id}', person_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, (255,255,255), thickness+1, cv2.LINE_AA)
            #     cv2.putText(img, f'person {person_id}', person_label_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize+0.2, c, thickness, cv2.LINE_AA)

            #     ang_label_line = 1  # Init line counter for the angle list

            # Angle lines, names, and values
            # print(f"Writing angles on the image for person {person_id}")

            for k, ang in enumerate(angles):
                if not np.isnan(ang):
                    ang_name = angle_names[k]
                    # print(f"Processing angle: {ang_name}")

                    if ang_name in relevant_angle_names:  # Only process relevant angles
                        ang_params = angle_dict.get(ang_name)
                        if ang_params is None:
                            # print(f"Angle {ang_name} not found in angle_dict!")
                            continue

                        # print(f"Angle parameters: {ang_params}")
                        keypoints = ang_params[0]
                        # print(f"Keypoints: {keypoints}")

                        if not any(kpt not in keypoints_names for kpt in keypoints):
                            ang_coords = np.array([
                                [X[keypoints_ids[keypoints_names.index(kpt)]], Y[keypoints_ids[keypoints_names.index(kpt)]]]
                                for kpt in keypoints if kpt in keypoints_names
                            ])

                            # Draw angle
                            if len(ang_coords) == 2:  # Segment angle
                                app_point, vec = draw_segment_angle(img, ang_coords, flip=1)
                                write_angle_on_body(img, ang, app_point, vec, np.array([1, 0]), dist=20, color=(255, 255, 255), fontSize=fontSize, thickness=thickness)
                            else:  # Joint angle
                                app_point, vec1, vec2 = draw_joint_angle(img, ang_coords, flip=1, right_angle=False)
                                write_angle_on_body(img, ang, app_point, vec1, vec2, dist=40, color=(0, 255, 0), fontSize=fontSize, thickness=thickness)

                        # if 'list' in display_angle_values_on and rom_data:
                        #     rom_fields = [
                        #         f"Test: {rom_data.get('test', '')}",
                        #         f"Ready: {rom_data.get('is_ready', False)}",
                        #         f"Trunk Angle: {rom_data.get('trunk', 'N/A')}",
                        #         f"ROM: {rom_data.get('ROM', ['N/A', 'N/A'])}",
                        #         f"ROM Range: {rom_data.get('rom_range', 'N/A')}",
                        #         f"Valid Position: {rom_data.get('position_valid', False)}",
                        #         f"Guidance: {rom_data.get('guidance', '')}",
                        #         f"Posture: {rom_data.get('posture_message', '')}",
                        #         f"Progress: {rom_data.get('ready_progress', 0)}%",
                        #         f"Status: {rom_data.get('status', '')}"
                        #     ]

    # for line, text in enumerate(rom_fields, start=1):
    #     text_position = (person_label_position[0], person_label_position[1] + int(line * 40 * fontSize))
    #     cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,0,0), thickness+1, cv2.LINE_AA)
    #     cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, c, thickness, cv2.LINE_AA)

    return img

def draw_segment_angle(img, ang_coords, flip):
    '''
    Draw a segment angle on the image.

    INPUTS:
    - img: opencv image
    - ang_coords: np.array. The 2D coordinates of the keypoints
    - flip: int. Whether the angle should be flipped

    OUTPUT:
    - app_point: np.array. The point where the angle is displayed
    - unit_segment_direction: np.array. The unit vector of the segment direction
    - img: image with the angle
    '''
    
    if not np.any(np.isnan(ang_coords)):
        app_point = np.int32(np.mean(ang_coords, axis=0))

        # segment line
        segment_direction = np.int32(ang_coords[0]) - np.int32(ang_coords[1])
        if (segment_direction==0).all():
            return app_point, np.array([0,0])
        unit_segment_direction = segment_direction/np.linalg.norm(segment_direction)
        cv2.line(img, app_point, np.int32(app_point+unit_segment_direction*20), (255,255,255), thickness)

        # horizontal line
        cv2.line(img, app_point, (np.int32(app_point[0])+flip*20, np.int32(app_point[1])), (255,255,255), thickness)

        return app_point, unit_segment_direction


def draw_joint_angle(img, ang_coords, flip, right_angle):
    '''
    Draw a joint angle on the image.

    INPUTS:
    - img: opencv image
    - ang_coords: np.array. The 2D coordinates of the keypoints
    - flip: int. Whether the angle should be flipped
    - right_angle: bool. Whether the angle should be offset by 90 degrees

    OUTPUT:
    - app_point: np.array. The point where the angle is displayed
    - unit_segment_direction: np.array. The unit vector of the segment direction
    - unit_parentsegment_direction: np.array. The unit vector of the parent segment direction
    - img: image with the angle
    '''
    
    if not np.any(np.isnan(ang_coords)):
        app_point = np.int32(ang_coords[1])
        
        segment_direction = np.int32(ang_coords[0] - ang_coords[1])
        parentsegment_direction = np.int32(ang_coords[-2] - ang_coords[-1])
        if (segment_direction==0).all() or (parentsegment_direction==0).all():
            return app_point, np.array([0,0]), np.array([0,0])
        
        if right_angle:
            segment_direction = np.array([-flip*segment_direction[1], flip*segment_direction[0]])
            segment_direction, parentsegment_direction = parentsegment_direction, segment_direction

        # segment line
        unit_segment_direction = segment_direction/np.linalg.norm(segment_direction)
        cv2.line(img, app_point, np.int32(app_point+unit_segment_direction*40), (0,255,0), thickness)
        
        # parent segment dotted line
        unit_parentsegment_direction = parentsegment_direction/np.linalg.norm(parentsegment_direction)
        draw_dotted_line(img, app_point, unit_parentsegment_direction, 40, color=(0, 255, 0), gap=7, dot_length=3, thickness=thickness)

        # arc
        start_angle = np.degrees(np.arctan2(unit_segment_direction[1], unit_segment_direction[0]))
        end_angle = np.degrees(np.arctan2(unit_parentsegment_direction[1], unit_parentsegment_direction[0]))
        if abs(end_angle - start_angle) > 180:
            if end_angle > start_angle: start_angle += 360
            else: end_angle += 360
        cv2.ellipse(img, app_point, (20, 20), 0, start_angle, end_angle, (0, 255, 0), thickness)

        return app_point, unit_segment_direction, unit_parentsegment_direction


def write_angle_on_body(img, ang, app_point, vec1, vec2, dist=40, color=(255,255,255), fontSize=0.3, thickness=1):
    '''
    Write the angle on the body.

    INPUTS:
    - img: opencv image
    - ang: float. The angle value to display
    - app_point: np.array. The point where the angle is displayed
    - vec1: np.array. The unit vector of the first segment
    - vec2: np.array. The unit vector of the second segment
    - dist: int. The distance from the origin where to write the angle
    - color: tuple. The color of the angle

    OUTPUT:
    - img: image with the angle
    '''

    vec_sum = vec1 + vec2
    if (vec_sum == 0.).all():
        return
    unit_vec_sum = vec_sum/np.linalg.norm(vec_sum)
    text_position = np.int32(app_point + unit_vec_sum*dist)
    cv2.putText(img, f'{ang:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, f'{ang:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness, cv2.LINE_AA)

def draw_dotted_line(img, start, direction, length, color=(0, 255, 0), gap=7, dot_length=3, thickness=thickness):
    '''
    Draw a dotted line with on a cv2 image

    INPUTS:
    - img: opencv image
    - start: np.array. The starting point of the line
    - direction: np.array. The direction of the line
    - length: int. The length of the line
    - color: tuple. The color of the line
    - gap: int. The distance between each dot
    - dot_length: int. The length of each dot
    - thickness: int. The thickness of the line

    OUTPUT:
    - img: image with the dotted line
    '''

    for i in range(0, length, gap):
        line_start = start + direction * i
        line_end = line_start + direction * dot_length
        cv2.line(img, tuple(line_start.astype(int)), tuple(line_end.astype(int)), color, thickness)