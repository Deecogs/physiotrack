from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import json
import tempfile
import uuid
import shutil
from pathlib import Path
import asyncio
import logging
import cv2
import numpy as np
from typing import List, Dict, Optional, Union
import time

# Import PhysioTrack
from physiotrack import PhysioTrack
from physiotrack.process import read_rom_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("physiotrack-api")

# Create FastAPI app
app = FastAPI(
    title="PhysioTrack ROM API",
    description="API for analyzing range of motion using the PhysioTrack library",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for uploads and results
TEMP_DIR = Path(tempfile.gettempdir()) / "physiotrack_api"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# Model for response
class ROMResponse(BaseModel):
    session_id: str
    status: str
    message: str
    rom_data: Optional[Dict] = None
    final_rom: Optional[Dict] = None
    video_path: Optional[str] = None

# Model for webcam settings
class WebcamSettings(BaseModel):
    height: float = 1.7
    visible_side: str = "auto"
    multiperson: bool = False
    mode: str = "performance"
    joint_angles: List[str] = ["Right knee", "Left knee", "Right hip", "Left hip", "Right shoulder", "Left shoulder"]
    segment_angles: List[str] = ["Right thigh", "Left thigh", "Trunk"]
    test_name: str = "lower_back_flexion"

# Store active webcam sessions
active_sessions = {}

# Helper function to create PhysioTrack config
def create_config(video_path, result_dir, options, is_webcam=False):
    """Create a configuration dictionary for PhysioTrack."""
    return {
        'project': {
            'video_input': ['webcam'] if is_webcam else [str(video_path)],
            'px_to_m_person_height': options.get('height', 1.7),
            'visible_side': [options.get('visible_side', 'auto')],
            'time_range': options.get('time_range', []),
            'webcam_id': options.get('webcam_id', 0),
            'input_size': [1280, 720],
            'load_trc_px': '',
            'compare': False
        },
        'process': {
            'multiperson': options.get('multiperson', False),
            'show_realtime_results': False,  # Always False for API
            'save_vid': True,
            'save_img': False,
            'save_pose': True,
            'calculate_angles': True,
            'save_angles': True,
            'result_dir': str(result_dir)
        },
        'pose': {
            'pose_model': options.get('pose_model', 'body_with_feet'),
            'mode': options.get('mode', 'performance'),
            'det_frequency': options.get('det_frequency', 4),
            'device': 'auto',
            'backend': 'auto',
            'tracking_mode': 'physiotrack',
            'keypoint_likelihood_threshold': 0.3,
            'average_likelihood_threshold': 0.5,
            'keypoint_number_threshold': 0.3,
            'slowmo_factor': 1
        },
        'angles': {
            'joint_angles': options.get('joint_angles', ['Right knee', 'Left knee', 'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder']),
            'segment_angles': options.get('segment_angles', ['Right thigh', 'Left thigh', 'Trunk']),
            'display_angle_values_on': ['body', 'list'],
            'fontSize': 0.3,
            'flip_left_right': True,
            'correct_segment_angles_with_floor_angle': True,
            'test_name': options.get('test_name', 'lower_back_flexion')
        },
        'px_to_meters_conversion': {
            'to_meters': True,
            'make_c3d': False,
            'calib_file': '',
            'floor_angle': 'auto',
            'xy_origin': ['auto'],
            'save_calib': True
        },
        'post-processing': {
            'interpolate': True,
            'filter': True,
            'show_graphs': False,
            'filter_type': options.get('filter_type', 'butterworth'),
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
            'participant_mass': [67.0],
            'right_left_symmetry': True,
            'default_height': options.get('height', 1.7),
            'fastest_frames_to_remove_percent': 0.1,
            'close_to_zero_speed_px': 50,
            'close_to_zero_speed_m': 0.2,
            'large_hip_knee_angles': 45,
            'trimmed_extrema_percent': 0.5,
            'osim_setup_path': '../OpenSim_setup'
        },
        'logging': {
            'use_custom_logging': True
        }
    }

# Process video file and get ROM data
async def process_video(video_path, session_id, options):
    logger.info(f"Processing video for session {session_id}")
    
    # Create results directory
    results_dir = TEMP_DIR / f"results_{session_id}"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Create configuration
        config = create_config(video_path, results_dir, options)
        
        # Process video with PhysioTrack
        PhysioTrack.process(config)
        
        # Find the ROM data file
        subfolder = f"{video_path.stem}_PhysioTrack"
        full_results_dir = results_dir / subfolder
        
        rom_files = list(full_results_dir.glob("*_rom_data.json"))
        
        if not rom_files:
            return {
                "status": "error",
                "message": "No ROM data files found. Processing may have failed.",
                "rom_data": None,
                "final_rom": None
            }
        
        # Read the ROM data
        rom_file = rom_files[0]
        rom_data = read_rom_data(str(rom_file))
        
        # Extract final ROM results
        final_rom = {}
        for angle_name, data in rom_data.items():
            # Get the last timepoint (highest time value) for final results
            last_timepoint = max(rom_data.keys(), key=float)
            final_rom = {
                "test_name": rom_data[last_timepoint]["test"],
                "ROM": rom_data[last_timepoint]["ROM"],
                "rom_range": rom_data[last_timepoint]["rom_range"],
                "angles": rom_data[last_timepoint]["angles"]
            }
            break
        
        # Get processed video path for frontend display
        processed_video = list(full_results_dir.glob("*.mp4"))
        video_path_str = str(processed_video[0]) if processed_video else None
        
        return {
            "status": "success", 
            "message": "Video processed successfully",
            "rom_data": rom_data,
            "final_rom": final_rom,
            "video_path": video_path_str
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error processing video: {str(e)}",
            "rom_data": None,
            "final_rom": None
        }

@app.post("/upload-video/", response_model=ROMResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    height: float = 1.7,
    visible_side: str = "auto",
    multiperson: bool = False,
    mode: str = "performance",
    test_name: str = "lower_back_flexion"
):
    """
    Upload and process a video file to analyze range of motion.
    
    - **file**: Video file (mp4, avi, mov)
    - **height**: Subject height in meters
    - **visible_side**: Visible side (auto, front, back, left, right, none)
    - **multiperson**: Whether to track multiple persons
    - **mode**: Processing mode (performance, balanced, lightweight)
    - **test_name**: Name of the test being performed
    """
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = TEMP_DIR / f"{session_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Configure options
    options = {
        "height": height,
        "visible_side": visible_side,
        "multiperson": multiperson,
        "mode": mode,
        "joint_angles": ["Right knee", "Left knee", "Right hip", "Left hip", "Right shoulder", "Left shoulder"],
        "segment_angles": ["Right thigh", "Left thigh", "Trunk"],
        "test_name": test_name
    }
    
    # Start processing in the background
    background_tasks.add_task(process_video, file_path, session_id, options)
    
    return {
        "session_id": session_id,
        "status": "processing",
        "message": "Video uploaded and processing started",
        "rom_data": None,
        "final_rom": None
    }

@app.get("/status/{session_id}", response_model=ROMResponse)
async def get_status(session_id: str):
    """
    Get the processing status and results for a session.
    
    - **session_id**: ID of the processing session
    """
    results_dir = TEMP_DIR / f"results_{session_id}"
    
    if not results_dir.exists():
        return {
            "session_id": session_id,
            "status": "processing",
            "message": "Processing in progress",
            "rom_data": None,
            "final_rom": None
        }
    
    # Find the ROM data files
    rom_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_rom_data.json"):
                rom_files.append(Path(root) / file)
    
    if not rom_files:
        return {
            "session_id": session_id,
            "status": "processing",
            "message": "Processing in progress, ROM data not yet available",
            "rom_data": None,
            "final_rom": None
        }
    
    # Read the ROM data
    rom_file = rom_files[0]
    rom_data = read_rom_data(str(rom_file))
    
    # Extract final ROM results
    final_rom = {}
    if rom_data:
        # Get the last timepoint (highest time value) for final results
        last_timepoint = max(rom_data.keys(), key=float)
        final_rom = {
            "test_name": rom_data[last_timepoint]["test"],
            "ROM": rom_data[last_timepoint]["ROM"],
            "rom_range": rom_data[last_timepoint]["rom_range"],
            "angles": rom_data[last_timepoint]["angles"]
        }
    
    # Find processed video for frontend display
    processed_video = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".mp4"):
                processed_video.append(Path(root) / file)
    
    video_path_str = str(processed_video[0]) if processed_video else None
    
    return {
        "session_id": session_id,
        "status": "completed",
        "message": "Processing completed",
        "rom_data": rom_data,
        "final_rom": final_rom,
        "video_path": video_path_str
    }

@app.websocket("/webcam/")
async def webcam_stream(websocket: WebSocket):
    """WebSocket endpoint for webcam-based ROM analysis"""
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {"status": "initializing"}
    
    try:
        # Receive webcam settings
        settings_json = await websocket.receive_text()
        settings = WebcamSettings(**json.loads(settings_json))
        
        # Create results directory
        results_dir = TEMP_DIR / f"webcam_{session_id}"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Configure PhysioTrack
        config = create_config(None, results_dir, {
            "height": settings.height,
            "visible_side": settings.visible_side,
            "multiperson": settings.multiperson,
            "mode": settings.mode,
            "joint_angles": settings.joint_angles,
            "segment_angles": settings.segment_angles,
            "test_name": settings.test_name,
            "webcam_id": 0
        }, is_webcam=True)
        
        # Start PhysioTrack in a separate process or thread
        # For simplicity, I'll show an example with a background task
        active_sessions[session_id] = {"status": "running", "config": config}
        
        # Send initial status
        await websocket.send_json({
            "session_id": session_id,
            "status": "running",
            "message": "Webcam ROM analysis started"
        })
        
        # Run PhysioTrack (this would normally be in a separate thread/process)
        # This is a simplified example - in production you would handle this differently
        PhysioTrack.process(config)
        
        # Find the ROM data file
        rom_files = list(results_dir.glob("**/*_rom_data.json"))
        
        if rom_files:
            # Read the ROM data
            rom_file = rom_files[0]
            rom_data = read_rom_data(str(rom_file))
            
            # Extract final ROM results
            final_rom = {}
            if rom_data:
                # Get the last timepoint for final results
                last_timepoint = max(rom_data.keys(), key=float)
                final_rom = {
                    "test_name": rom_data[last_timepoint]["test"],
                    "ROM": rom_data[last_timepoint]["ROM"],
                    "rom_range": rom_data[last_timepoint]["rom_range"],
                    "angles": rom_data[last_timepoint]["angles"]
                }
            
            # Find processed video
            processed_videos = list(results_dir.glob("**/*.mp4"))
            video_path_str = str(processed_videos[0]) if processed_videos else None
            
            # Send results
            await websocket.send_json({
                "session_id": session_id,
                "status": "completed",
                "message": "Webcam analysis completed",
                "rom_data": rom_data,
                "final_rom": final_rom,
                "video_path": video_path_str
            })
        else:
            await websocket.send_json({
                "session_id": session_id,
                "status": "error",
                "message": "No ROM data generated. Analysis may have failed."
            })
            
    except Exception as e:
        logger.error(f"Error in webcam stream: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "session_id": session_id,
                "status": "error",
                "message": f"Error: {str(e)}"
            })
        except:
            pass
    finally:
        # Clean up
        if session_id in active_sessions:
            del active_sessions[session_id]
        await websocket.close()

# Static files for the web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint returns the web interface
@app.get("/")
async def get_root():
    return {"message": "PhysioTrack ROM API is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)