import toml
from pathlib import Path
from threading import Thread
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from queue import Queue
import cv2
import json
import time
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
        f"""
        <html>
        <head>
            <title>Webcam Stream</title>
            <style>
                .container {{
                    display: flex;
                    justify-content: center;
                    align-items: flex-start;
                    gap: 20px;
                    padding: 20px;
                }}
                .video-container {{
                    flex: 2;
                    min-width: 800px;
                }}
                .data-container {{
                    flex: 1;
                    min-width: 300px;
                    border: 1px solid #ccc;
                    padding: 15px;
                    border-radius: 5px;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .data-item {{
                    margin-bottom: 15px;
                    padding: 12px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                }}
                .data-label {{
                    font-weight: bold;
                    color: #6c757d;
                    margin-bottom: 8px;
                    display: block;
                }}
                .data-value {{
                    color: #212529;
                }}
                .angles-container, .rom-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-top: 8px;
                }}
                .angle-item, .rom-item {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 4px;
                }}
                .angle-label, .rom-label {{
                    font-weight: bold;
                    color: #6c757d;
                }}
                .angle-value, .rom-value {{
                    color: #212529;
                }}
                img {{
                    border: 1px solid black;
                    width: 100%;
                    height: auto;
                    max-height: 720px;
                }}
                .text-center {{
                    text-align: center;
                }}
                .text-muted {{
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <h1>Webcam Stream</h1>
            <div class="container">
                <div class="video-container">
                    <h3>Processed Video</h3>
                    <img src="/video_feed" />
                </div>
                <div class="data-container">
                    <h3>Real-time Data</h3>
                    <div id="realTimeData">
                        <div class="text-center text-muted">Waiting for data...</div>
                    </div>
                </div>
            </div>
            <script>
                // Function to update real-time data display
                function updateRealTimeData() {{
                    const eventSource = new EventSource('/json_feed');
                    
                    eventSource.onmessage = function(event) {{
                        try {{
                            const data = JSON.parse(event.data);
                            const realTimeDataDiv = document.getElementById('realTimeData');
                            realTimeDataDiv.innerHTML = '';
                            
                            // Display frame data
                            if (data.frame_data) {{
                                const frameDiv = document.createElement('div');
                                frameDiv.className = 'data-item';
                                frameDiv.innerHTML = `
                                    <div class="data-label">Frame Number:</div>
                                    <div class="data-value">${{data.frame_data.frame_number}}</div>
                                `;
                                realTimeDataDiv.appendChild(frameDiv);
                            }}
                            
                            // Display angles in a more readable format
                            if (data.angles) {{
                                const anglesDiv = document.createElement('div');
                                anglesDiv.className = 'data-item';
                                anglesDiv.innerHTML = `
                                    <div class="data-label">Angles:</div>
                                    <div class="data-value">
                                        <div class="angles-container">
                                            ${{
                                                Object.entries(data.angles).map(([key, value]) => `
                                                    <div class="angle-item">
                                                        <span class="angle-label">${{key}}:</span>
                                                        <span class="angle-value">${{value.toFixed(1)}}°</span>
                                                    </div>
                                                `).join('')
                                            }}
                                        </div>
                                    </div>
                                `;
                                realTimeDataDiv.appendChild(anglesDiv);
                            }}
                            
                            // Display ROM
                            if (data.rom) {{
                                const romDiv = document.createElement('div');
                                romDiv.className = 'data-item';
                                romDiv.innerHTML = `
                                    <div class="data-label">ROM:</div>
                                    <div class="data-value">
                                        <div class="rom-container">
                                            ${{
                                                Object.entries(data.rom).map(([key, value]) => `
                                                    <div class="rom-item">
                                                        <span class="rom-label">${{key}}:</span>
                                                        <span class="rom-value">${{value.toFixed(1)}}°</span>
                                                    </div>
                                                `).join('')
                                            }}
                                        </div>
                                    </div>
                                `;
                                realTimeDataDiv.appendChild(romDiv);
                            }}
                        }} catch (error) {{
                            console.error('Error processing real-time data:', error);
                        }}
                    }};
                    
                    eventSource.onerror = function() {{
                        console.error('EventSource connection error');
                        eventSource.close();
                        // Reconnect after 1 second
                        setTimeout(updateRealTimeData, 1000);
                    }};
                }}

                // Start updating real-time data when the page loads
                window.onload = function() {{
                    updateRealTimeData();
                }};
            </script>
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
                    align-items: flex-start;
                    gap: 20px;
                    padding: 20px;
                }}
                .video-container {{
                    flex: 2;
                    min-width: 800px;
                }}
                .data-container {{
                    flex: 1;
                    min-width: 300px;
                    border: 1px solid #ccc;
                    padding: 15px;
                    border-radius: 5px;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .data-item {{
                    margin-bottom: 15px;
                    padding: 12px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                }}
                .data-label {{
                    font-weight: bold;
                    color: #6c757d;
                    margin-bottom: 8px;
                    display: block;
                }}
                .data-value {{
                    color: #212529;
                }}
                .angles-container, .rom-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-top: 8px;
                }}
                .angle-item, .rom-item {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 4px;
                }}
                .angle-label, .rom-label {{
                    font-weight: bold;
                    color: #6c757d;
                }}
                .angle-value, .rom-value {{
                    color: #212529;
                }}
                img {{
                    border: 1px solid black;
                    width: 100%;
                    height: auto;
                    max-height: 720px;
                }}
                .text-center {{
                    text-align: center;
                }}
                .text-muted {{
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <h1>Video Analysis: {filename}</h1>
            <div class="container">
                <div class="video-container">
                    <h3>Processed Video</h3>
                    <img src="/video_feed" />
                </div>
                <div class="data-container">
                    <h3>Real-time Data</h3>
                    <div id="realTimeData">
                        <div class="text-center text-muted">Waiting for data...</div>
                    </div>
                </div>
            </div>
            <script>
                // Function to update real-time data display
                function updateRealTimeData() {{
                    const eventSource = new EventSource('/json_feed');
                    
                    eventSource.onmessage = function(event) {{
                        try {{
                            const data = JSON.parse(event.data);
                            const realTimeDataDiv = document.getElementById('realTimeData');
                            realTimeDataDiv.innerHTML = '';
                            
                            // Display frame data
                            if (data.frame_data) {{
                                const frameDiv = document.createElement('div');
                                frameDiv.className = 'data-item';
                                frameDiv.innerHTML = `
                                    <div class="data-label">Frame Number:</div>
                                    <div class="data-value">${{data.frame_data.frame_number}}</div>
                                `;
                                realTimeDataDiv.appendChild(frameDiv);
                            }}
                            
                            // Display angles in a more readable format
                            if (data.angles) {{
                                const anglesDiv = document.createElement('div');
                                anglesDiv.className = 'data-item';
                                anglesDiv.innerHTML = `
                                    <div class="data-label">Angles:</div>
                                    <div class="data-value">
                                        <div class="angles-container">
                                            ${{
                                                Object.entries(data.angles).map(([key, value]) => `
                                                    <div class="angle-item">
                                                        <span class="angle-label">${{key}}:</span>
                                                        <span class="angle-value">${{value.toFixed(1)}}°</span>
                                                    </div>
                                                `).join('')
                                            }}
                                        </div>
                                    </div>
                                `;
                                realTimeDataDiv.appendChild(anglesDiv);
                            }}
                            
                            // Display ROM
                            if (data.rom) {{
                                const romDiv = document.createElement('div');
                                romDiv.className = 'data-item';
                                romDiv.innerHTML = `
                                    <div class="data-label">ROM:</div>
                                    <div class="data-value">
                                        <div class="rom-container">
                                            ${{
                                                Object.entries(data.rom).map(([key, value]) => `
                                                    <div class="rom-item">
                                                        <span class="rom-label">${{key}}:</span>
                                                        <span class="rom-value">${{value.toFixed(1)}}°</span>
                                                    </div>
                                                `).join('')
                                            }}
                                        </div>
                                    </div>
                                `;
                                realTimeDataDiv.appendChild(romDiv);
                            }}
                        }} catch (error) {{
                            console.error('Error processing real-time data:', error);
                        }}
                    }};
                    
                    eventSource.onerror = function() {{
                        console.error('EventSource connection error');
                        eventSource.close();
                        // Reconnect after 1 second
                        setTimeout(updateRealTimeData, 1000);
                    }};
                }}

                // Start updating real-time data when the page loads
                window.onload = function() {{
                    updateRealTimeData();
                }};
            </script>
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

@app.get("/json_feed")
def json_feed():
    def gen():
        # Get the results directory
        results_dir = Path(__file__).parent / "results"
        
        while True:
            try:
                # Refresh the list of stream files
                stream_files = list(results_dir.glob("**/*stream*.json"))
                
                if stream_files:
                    # Get the latest stream file
                    stream_file = max(stream_files, key=lambda p: p.stat().st_mtime)
                    
                    # Try to read the file with a small timeout
                    try:
                        with open(stream_file, 'r') as f:
                            data = json.load(f)
                            # Get the latest frame data
                            latest_frame = max(data.keys())
                            frame_data = data[latest_frame]
                            
                            # Convert to JSON string
                            json_str = json.dumps(frame_data)
                            yield f"data: {json_str}\n\n"
                    except (json.JSONDecodeError, OSError) as e:
                        # If file is being written to, wait and try again
                        yield f"data: {{\"warning\": \"File is being updated\"}}\n\n"
                        time.sleep(0.1)
                        continue
                else:
                    yield f"data: {{\"error\": \"No stream files found\"}}\n\n"
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                time.sleep(1)

    return StreamingResponse(gen(), media_type='text/event-stream')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
