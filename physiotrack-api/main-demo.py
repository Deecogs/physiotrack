from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import os
import json
import shutil
from pathlib import Path
import tempfile
import uuid
import logging

# Import PhysioTrack
from physiotrack import PhysioTrack
from physiotrack.process import read_rom_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("physiotrack-api")

app = FastAPI(title="PhysioTrack ROM Analysis API", 
              description="API for processing videos with PhysioTrack and returning ROM data")

# Create directories with absolute paths
BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
TEMP_DIR = BASE_DIR / "temp_uploads"
RESULTS_DIR = BASE_DIR / "results"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Create HTML template file
def create_template_file():
    template_path = TEMPLATES_DIR / "upload.html"
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhysioTrack Video Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
        }
        input[type="file"],
        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        input[type="file"] {
            padding: 10px 0;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
            display: block;
            width: 100%;
            max-width: 300px;
            margin: 20px auto 0;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        .progress-container {
            margin-top: 20px;
            display: none;
        }
        .progress-bar {
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            margin-bottom: 10px;
            overflow: hidden;
        }
        .progress {
            height: 100%;
            background-color: #3498db;
            width: 0%;
            transition: width 0.3s ease;
        }
        .status {
            text-align: center;
            font-weight: 600;
        }
        .result-container {
            margin-top: 30px;
            display: none;
        }
        .result-video {
            max-width: 100%;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .rom-summary {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        .error-message {
            background-color: #fee;
            color: #c0392b;
            padding: 10px;
            border-radius: 4px;
            margin-top: 20px;
            display: none;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f0f7ff;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        code {
            font-family: monospace;
        }
    </style>
</head>
<body>
    <h1>PhysioTrack Video Analysis</h1>
    
    <div class="container">
        <div class="instructions">
            <h3>Instructions</h3>
            <p>Upload a video to analyze range of motion (ROM) data using PhysioTrack:</p>
            <ol>
                <li>Select a video file (MP4 format recommended)</li>
                <li>Enter the person's height in meters (used for scaling)</li>
                <li>Click "Process Video" and wait for the analysis to complete</li>
            </ol>
            <p>The system will process the video and return ROM data, a processed video with overlays, and a summary of the range of motion.</p>
        </div>

        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="videoFile">Video File:</label>
                <input type="file" id="videoFile" name="file" accept="video/*" required>
            </div>
            
            <div class="form-group">
                <label for="heightMeters">Person's Height (meters):</label>
                <input type="number" id="heightMeters" name="height_meters" step="0.01" min="0.5" max="2.5" value="1.70" required>
            </div>
            
            <button type="submit" id="submitBtn">Process Video</button>
        </form>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar">
                <div class="progress" id="progressBar"></div>
            </div>
            <div class="status" id="status">Processing video...</div>
        </div>
        
        <div class="error-message" id="errorMessage"></div>
    </div>
    
    <div class="container result-container" id="resultContainer">
        <h2>Analysis Results</h2>
        
        <h3>Processed Video</h3>
        <video class="result-video" id="resultVideo" controls></video>
        
        <h3>ROM Summary</h3>
        <div class="rom-summary" id="romSummary"></div>
        
        <h3>Full ROM Data</h3>
        <pre id="romData"><code>Loading ROM data...</code></pre>
    </div>
    
    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const form = new FormData(this);
            const submitBtn = document.getElementById('submitBtn');
            const progressContainer = document.getElementById('progressContainer');
            const progressBar = document.getElementById('progressBar');
            const status = document.getElementById('status');
            const errorMessage = document.getElementById('errorMessage');
            const resultContainer = document.getElementById('resultContainer');
            
            // Reset UI
            errorMessage.style.display = 'none';
            resultContainer.style.display = 'none';
            
            // Show progress
            submitBtn.disabled = true;
            progressContainer.style.display = 'block';
            status.textContent = 'Uploading video...';
            progressBar.style.width = '10%';
            
            try {
                // Simulate progress while waiting for the server
                let progress = 10;
                const progressInterval = setInterval(() => {
                    if (progress < 90) {
                        progress += 2;
                        progressBar.style.width = progress + '%';
                        if (progress > 20) {
                            status.textContent = 'Processing video with PhysioTrack...';
                        }
                    }
                }, 1000);
                
                // Send the form data
                const response = await fetch('/process-video/', {
                    method: 'POST',
                    body: form
                });
                
                clearInterval(progressInterval);
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Error processing video');
                }
                
                // Complete progress bar
                progressBar.style.width = '100%';
                status.textContent = 'Processing complete!';
                
                // Get the data
                const data = await response.json();
                const sessionId = data.session_id;
                
                // Display the results
                resultContainer.style.display = 'block';
                
                // Set video source
                const resultVideo = document.getElementById('resultVideo');
                resultVideo.src = `/results/${sessionId}/video`;
                
                // Get ROM summary
                const summaryResponse = await fetch(`/results/${sessionId}/summary`);
                const summaryData = await summaryResponse.json();
                
                // Display ROM summary
                const romSummary = document.getElementById('romSummary');
                let summaryHTML = '<table style="width:100%; border-collapse: collapse;">';
                summaryHTML += '<tr><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Joint</th>' +
                               '<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Min (°)</th>' +
                               '<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Max (°)</th>' +
                               '<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">ROM (°)</th></tr>';
                
                for (const [joint, values] of Object.entries(summaryData)) {
                    summaryHTML += `<tr>
                        <td style="border: 1px solid #ddd; padding: 8px;">${joint}</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">${values.min.toFixed(1)}</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">${values.max.toFixed(1)}</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">${values.rom.toFixed(1)}</td>
                    </tr>`;
                }
                summaryHTML += '</table>';
                romSummary.innerHTML = summaryHTML;
                
                // Display full ROM data
                const romData = document.getElementById('romData');
                romData.textContent = JSON.stringify(data.rom_data, null, 2);
                
            } catch (error) {
                console.error('Error:', error);
                errorMessage.textContent = error.message || 'An error occurred during processing';
                errorMessage.style.display = 'block';
                progressBar.style.width = '0%';
                status.textContent = 'Processing failed';
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""
    
    with open(template_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Created HTML template at {template_path}")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting PhysioTrack API")
    # Create necessary directories if they don't exist
    TEMP_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(exist_ok=True)
    
    # Create the HTML template file
    create_template_file()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down PhysioTrack API")
    # Clean up temp files (optional, uncomment if needed)
    # shutil.rmtree(TEMP_DIR, ignore_errors=True)

def process_video(video_path: Path, session_id: str, height_meters: float = 1.70):
    """Process the video with PhysioTrack and return the ROM data path."""
    logger.info(f"Processing video: {video_path}")
    
    # Make sure video_path is absolute
    video_path = Path(os.path.abspath(video_path))
    
    if not video_path.exists():
        logger.error(f"Video file not found at {video_path}")
        raise FileNotFoundError(f"Video file not found at {video_path}")
    
    # Create a session-specific result directory
    result_dir = RESULTS_DIR / session_id
    result_dir.mkdir(exist_ok=True)
    
    # Get the absolute directory of the video file
    video_dir = video_path.parent
    
    # Create a configuration dictionary for PhysioTrack
    config_dict = {
        'project': {
            'video_input': [str(video_path)],  # Use absolute path
            'px_to_m_person_height': height_meters,
            'visible_side': ['auto'],
            'time_range': [],  # Analyze the whole video
            'video_dir': str(video_dir),  # Use absolute path of parent dir
            'webcam_id': 0,
            'input_size': [1280, 720],
            'load_trc_px': '',
            'compare': False
        },
        'process': {
            'multiperson': False,  # Focus on one person
            'show_realtime_results': False,  # No GUI in server mode
            'save_vid': True,
            'save_img': False,
            'save_pose': True,
            'calculate_angles': True,
            'save_angles': True,
            'result_dir': str(result_dir.absolute())  # Use absolute path
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
            'show_graphs': False,  # No GUI in server mode
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
            'default_height': height_meters,
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
            'use_custom_logging': True,  # Enable detailed logging
            'log_level': 'DEBUG'
        }
    }
    
    # Debug outputs
    logger.info(f"Video path absolute: {video_path}")
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Results directory: {result_dir.absolute()}")
    
    try:
        # Run PhysioTrack with the configuration
        PhysioTrack.process(config_dict)
        logger.info("PhysioTrack processing complete")
        
        # Get the video filename without extension to match PhysioTrack's output patterns
        video_name = video_path.stem
        subfolder = f"{video_name}_PhysioTrack"
        full_results_dir = result_dir / subfolder
        
        logger.info(f"Looking for ROM data in: {full_results_dir}")
        
        # Find the ROM data file
        rom_files = list(full_results_dir.glob("*_rom_data.json"))
        
        if not rom_files:
            logger.error("No ROM data files found. Processing may have failed.")
            # Try to find the file with a more general search
            rom_files = list(result_dir.glob("**/*_rom_data.json"))
            if not rom_files:
                raise Exception("ROM data generation failed")
        
        logger.info(f"Found ROM data file: {rom_files[0]}")
        return rom_files[0]
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the upload form."""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/process-video/", response_class=JSONResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    height_meters: Optional[float] = 1.70
):
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Create a temporary directory for this upload
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Save the uploaded file
    temp_video_path = session_dir / file.filename
    
    try:
        # Save uploaded file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Video uploaded to {temp_video_path}")
        
        # Process the video (this could take time)
        rom_data_path = process_video(temp_video_path, session_id, height_meters)
        
        # Read the ROM data
        rom_data = read_rom_data(str(rom_data_path))
        
        # Schedule cleanup in the background after response is sent
        # Only clean up temp uploads, not results
        background_tasks.add_task(cleanup_files, session_dir)
        
        # Return the ROM data as JSON
        return JSONResponse(
            content={
                "message": "Video processed successfully",
                "session_id": session_id,
                "rom_data": rom_data
            }
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        # Clean up on error
        if session_dir.exists():
            background_tasks.add_task(cleanup_files, session_dir)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/results/{session_id}/video")
async def get_processed_video(session_id: str):
    """Get the processed video with overlays."""
    result_dir = RESULTS_DIR / session_id
    
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find the processed video file (matches the naming pattern from PhysioTrack)
    video_files = list(result_dir.glob("**/*_PhysioTrack.mp4"))
    
    if not video_files:
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(path=str(video_files[0]), media_type="video/mp4")

@app.get("/results/{session_id}/summary")
async def get_rom_summary(session_id: str):
    """Get a summary of the ROM analysis."""
    result_dir = RESULTS_DIR / session_id
    
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find the ROM data file
    rom_files = list(result_dir.glob("**/*_rom_data.json"))
    
    if not rom_files:
        raise HTTPException(status_code=404, detail="ROM data not found")
    
    # Read the ROM data
    rom_data = read_rom_data(str(rom_files[0]))
    
    # Calculate the ROM summary (similar to the demo script)
    times = [float(t) for t in rom_data.keys()]
    times.sort()
    
    if not times:
        return JSONResponse(content={"error": "No time points found in ROM data"})
    
    # Get joint names from first time point
    first_time = str(times[0])
    if first_time not in rom_data or 'angles' not in rom_data[first_time]:
        return JSONResponse(content={"error": "Invalid ROM data format"})
    
    joint_names = list(rom_data[first_time]['angles'].keys())
    
    # Calculate min, max, and range for each joint
    summary_data = {}
    for joint in joint_names:
        # Extract angle data for this joint
        angles = []
        for t in times:
            time_str = str(t)
            if time_str in rom_data and 'angles' in rom_data[time_str] and joint in rom_data[time_str]['angles']:
                angles.append(rom_data[time_str]['angles'][joint])
        
        if angles:
            min_val = min(angles)
            max_val = max(angles)
            rom_range = max_val - min_val
            
            summary_data[joint] = {
                'min': min_val,
                'max': max_val,
                'rom': rom_range
            }
    
    return JSONResponse(content=summary_data)

def cleanup_files(directory: Path):
    """Clean up temporary files after processing."""
    try:
        if directory.exists():
            shutil.rmtree(directory)
            logger.info(f"Cleaned up directory: {directory}")
    except Exception as e:
        logger.error(f"Error cleaning up directory {directory}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)