import toml
from pathlib import Path
from threading import Thread
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from queue import Queue
import cv2
import json
from physiotrack import PhysioTrack
import physiotrack.process_webpage as pw
import physiotrack.process as proc

# Path to the demo config
# CONFIG_PATH = Path(__file__).parent / 'physiotrack' / 'Demo' / 'Config_demo.toml'
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

import physiotrack.process as proc
frame_queue = Queue(maxsize=1)  # shared MJPEG frame queue
# override both modules to use FastAPI queue instead of Flask server
pw.start_webpage_stream = proc.start_webpage_stream = lambda: frame_queue

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(
        """
        <html><head><title>PhysioTrack</title></head><body>
        <h1>PhysioTrack</h1>
        <ul>
          <li><a href="/webcam">Webcam Analysis</a></li>
          <li><a href="/upload">Upload Video</a></li>
        </ul>
        </body></html>
        """
    )

@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page():
    return HTMLResponse(
        """
        <html><head><title>Webcam Analysis</title></head><body>
        <h1>Webcam Analysis</h1>
        <button onclick="window.location.href='/start_webcam'">Start Analysis</button>
        </body></html>
        """
    )

@app.get("/start_webcam", response_class=HTMLResponse)
async def start_webcam():
    # config = toml.load(CONFIG_PATH)
    config = config_dict.copy()
    # override for webcam
    config['project']['video_input'] = ['webcam']
    config['process']['show_realtime_results'] = True
    config['process']['save_vid'] = False
    config['process']['save_img'] = False
    config['process']['save_pose'] = False
    config['process']['save_angles'] = True

    # start processing in background
    Thread(target=PhysioTrack.process, args=(config,), daemon=True).start()

    return HTMLResponse(
        """
        <html>
        <head>
            <title>Webcam Stream</title>
            <style>
                .container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 20px;
                }
                img {
                    border: 1px solid black;
                    width: 640px;
                    height: 480px;
                }
            </style>
        </head>
        <body>
        <h1>Webcam Stream</h1>
        <div class="container">
            <div>
                <h3>Processed Video</h3>
                <img src="/video_feed" />
            </div>
        </div>
        </body>
        </html>
        """
    )

# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    return HTMLResponse(
        """
        <html><head><title>Upload Video</title></head><body>
        <h1>Upload Video</h1>
        <form action="/upload" enctype="multipart/form-data" method="post">
          <input type="file" name="file" accept="video/*" />
          <button type="submit">Upload</button>
        </form>
        </body></html>
        """
    )

@app.post("/upload", response_class=HTMLResponse)
async def upload_video(file: UploadFile = File(...)):
    upload_dir = Path(__file__).parent / 'uploads'
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, 'wb') as f:
        f.write(await file.read())
    return HTMLResponse(
        f"""
        <html><head><title>Video Uploaded</title></head><body>
        <h1>Uploaded: {file.filename}</h1>
        <button onclick="window.location.href='/start_video?filename={file.filename}'">Start Analysis</button>
        </body></html>
        """
    )

@app.get("/start_video", response_class=HTMLResponse)
async def start_video(filename: str):
    file_path = Path(__file__).parent / 'uploads' / filename
    # config = toml.load(CONFIG_PATH)
    config = config_dict.copy()
    # override for uploaded video
    config['project']['video_input'] = [str(file_path)]
    config['process']['show_realtime_results'] = True
    config['process']['save_vid'] = False
    config['process']['save_img'] = False
    config['process']['save_pose'] = False
    config['process']['save_angles'] = True
    # start processing in background
    Thread(target=PhysioTrack.process, args=(config,), daemon=True).start()
    return HTMLResponse(
        f"""
        <html>
        <head>
            <title>Video Analysis</title>
            <style>
            .container {{
                display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 20px;
                }}
                img {{
                    border: 1px solid black;
                    width: 640px;
                    height: 480px;
                }}
            </style>
        </head>
        <body>
            <h1>Video Analysis: {filename}</h1>
            <div class="container">
                <div>
                    <h3>Processed Video</h3>
                    <img src="/video_feed" />
                </div>
            </div>
        </body>
        </html>
        """
    )

@app.get("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
