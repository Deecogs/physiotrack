import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import toml

from physiotrack.process_webpage import initialize_pose_tracker, process_frame, generate_rom_data

# Load config (adapt path as needed)
CONFIG_PATH = Path(__file__).parent / 'physiotrack' / 'Demo' / 'Config_demo.toml'
config_dict = toml.load(CONFIG_PATH)

# Initialize pose tracker and related objects once
pose_tracker, angle_dict, keypoints_ids, keypoints_names, model_root = initialize_pose_tracker(config_dict)

st.title("PhysioTrack Web Demo")

# Option to upload a video or use webcam
input_mode = st.radio("Select input mode:", ("Webcam", "Upload Video"))

if input_mode == "Webcam":
    import pandas as pd
    stframe = st.empty()
    stop = st.button("Stop Webcam")
    cap = cv2.VideoCapture(0)
    angle_records = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop:
            break
        processed_frame, angle_dict_example = process_frame(frame, pose_tracker, config_dict, angle_dict, keypoints_ids, keypoints_names, model_root)
        stframe.image(processed_frame, channels="BGR")
        fps = 30
        time_val = frame_idx / fps
        if angle_dict_example is not None:
            angle_records.append({"time": time_val, **angle_dict_example})
        frame_idx += 1
    cap.release()
    # --- ROM data generation (after loop) ---
    if angle_records:
        angle_df = pd.DataFrame(angle_records)
        from physiotrack.process_webpage import generate_rom_data
        rom_data = generate_rom_data(angle_df, "webcam_test", "webcam_rom.json")
        st.json(rom_data)
        with open("webcam_rom.json", "rb") as f:
            st.download_button("Download ROM Data (JSON)", f, file_name="webcam_rom.json")

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
        import pandas as pd
        angle_records = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, angle_dict_example = process_frame(frame, pose_tracker, config_dict, angle_dict, keypoints_ids, keypoints_names, model_root)
            stframe.image(processed_frame, channels="BGR")
            time_val = frame_idx / fps
            if angle_dict_example is not None:
                angle_records.append({"time": time_val, **angle_dict_example})
            frame_idx += 1
        cap.release()
        # --- ROM data generation (after loop) ---
        if angle_records:
            angle_df = pd.DataFrame(angle_records)
            rom_data = generate_rom_data(angle_df, "video_test", "video_rom.json")
            # Visualize ROM data for the first available angle
            if rom_data:
                first_time = next(iter(rom_data))
                angles = rom_data[first_time]["angles"].keys()
                if angles:
                    first_angle = next(iter(angles))
                    times = [float(t) for t in rom_data.keys()]
                    times.sort()
                    angle_vals = [rom_data[str(t)]["angles"][first_angle] for t in times]
                    import pandas as pd
                    chart_df = pd.DataFrame({"time": times, first_angle: angle_vals})
                    st.line_chart(chart_df.set_index("time"))
            st.json(rom_data)
            with open("video_rom.json", "rb") as f:
                st.download_button("Download ROM Data (JSON)", f, file_name="video_rom.json")
