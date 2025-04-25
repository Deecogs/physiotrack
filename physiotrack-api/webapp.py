import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import toml

from physiotrack.process_webpage import initialize_pose_tracker, process_frame, generate_rom_data

# Load config (adapt path as needed)
CONFIG_PATH = Path(__file__).parent / 'physiotrack' / 'Demo' / 'Config_demo.toml'
# config_dict = toml.load(CONFIG_PATH)
# Create a configuration dictionary for PhysioTrack
config_dict = {
    'project': {
        'video_input': ['webcam'],
        'px_to_m_person_height': 1.65,
        'visible_side': ['auto'],
        'time_range': [],  # Analyze the whole video
        # 'video_dir': str(demo_dir),  # Make sure this is a string
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
        'result_dir': 'results/webapp_PhysioTrack'
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

# Initialize pose tracker and related objects once
pose_tracker, angle_dict, keypoints_ids, keypoints_names, model_root = initialize_pose_tracker(config_dict)

st.title("PhysioTrack Web Demo")

# Option to upload a video or use webcam
input_mode = st.radio("Select input mode:", ("Webcam", "Upload Video"))

if input_mode == "Webcam":
    import pandas as pd
    stframe = st.empty()
    stjson = st.empty()
    stchart = st.empty()
    # Use session state to track webcam running state
    if 'webcam_running' not in st.session_state:
        st.session_state['webcam_running'] = False

    if not st.session_state['webcam_running']:
        if st.button("Start Webcam"):
            st.session_state['webcam_running'] = True
            if hasattr(st, 'experimental_rerun'):
                st.experimental_rerun()
    elif st.session_state['webcam_running']:
        if st.button("Stop Webcam"):
            st.session_state['webcam_running'] = False
            if hasattr(st, 'experimental_rerun'):
                st.experimental_rerun()
        else:
            cap = cv2.VideoCapture(0)
            angle_records = []
            frame_idx = 0
            running = True
            import time
            # Set up placeholders for left (processed) and right (original) video
            col1, col2 = st.columns(2)
            processed_placeholder = col1.empty()
            original_placeholder = col2.empty()
            while cap.isOpened() and running:
                ret, frame = cap.read()
                if not ret:
                    running = False
                    st.session_state['webcam_running'] = False
                    break
                processed_frame, angle_dict_example = process_frame(frame, pose_tracker, config_dict, angle_dict, keypoints_ids, keypoints_names, model_root)
                # Show processed frame on the LEFT, original on the RIGHT
                processed_placeholder.image(processed_frame, channels="BGR", caption="Processed (Live)")
                original_placeholder.image(frame, channels="BGR", caption="Webcam (Live)")
                # --- Show only current angles for each frame ---
                if 'angle_placeholder' not in locals():
                    angle_placeholder = st.empty()
                if angle_dict_example is not None:
                    angle_placeholder.markdown("### Current Angles")
                    angle_placeholder.json(angle_dict_example)
                frame_idx += 1
                time.sleep(1/30)
            cap.release()

elif input_mode == "Upload Video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if video_file is not None:
        import tempfile
        temp_dir = tempfile.gettempdir()
        tfile = Path(temp_dir) / video_file.name
        with open(tfile, 'wb') as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(str(tfile))
        stframe = st.empty()
        stjson = st.empty()
        stchart = st.empty()
        import pandas as pd
        angle_records = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        # Set up a placeholder for the current angles
        angle_placeholder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, angle_dict_example = process_frame(frame, pose_tracker, config_dict, angle_dict, keypoints_ids, keypoints_names, model_root)
            stframe.image(processed_frame, channels="BGR")
            if angle_dict_example is not None:
                angle_placeholder.markdown("### Current Angles")
                angle_placeholder.json(angle_dict_example)
            frame_idx += 1
        cap.release()
