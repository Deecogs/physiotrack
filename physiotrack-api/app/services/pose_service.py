import os
import json
import asyncio
from pathlib import Path
import logging
import traceback
from typing import Dict, Any

from physiotrack import PhysioTrack

logger = logging.getLogger(__name__)

async def process_video(video_path: str, output_dir: str, params: Dict[str, Any]):
    """
    Process a video using PhysioTrack
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the results
        params: Parameters for the processing
    """
    # Create status file
    status_file = Path(output_dir) / "status.json"
    with open(status_file, "w") as f:
        json.dump({
            "status": "processing",
            "message": "Starting video processing"
        }, f)
    
    try:
        # Prepare PhysioTrack configuration
        assessment_id = Path(output_dir).name
        config = {
            'project': {
                'video_input': [video_path],
                'px_to_m_person_height': params.height if hasattr(params, 'height') else 1.7,
                'visible_side': [params.visible_side] if hasattr(params, 'visible_side') else ['auto'],
                'time_range': params.time_range if hasattr(params, 'time_range') and params.time_range else []
            },
            'process': {
                'multiperson': False,  # Focus on one person for assessment
                'show_realtime_results': False,  # Headless operation
                'save_vid': True,
                'save_img': False,
                'save_pose': True,
                'calculate_angles': True,
                'save_angles': True,
                'result_dir': output_dir
            },
            'pose': {
                'pose_model': 'body_with_feet',
                'mode': 'balanced',
                'det_frequency': 4,
                'tracking_mode': 'physiotrack'
            },
            'angles': {
                'joint_angles': params.joint_angles if hasattr(params, 'joint_angles') else [
                    'Right ankle', 'Left ankle', 'Right knee', 'Left knee', 
                    'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder', 
                    'Right elbow', 'Left elbow'
                ],
                'segment_angles': params.segment_angles if hasattr(params, 'segment_angles') else [
                    'Right thigh', 'Left thigh', 'Trunk'
                ]
            },
            'post-processing': {
                'interpolate': True,
                'filter': True,
                'show_graphs': False
            }
        }
        
        # Process video with PhysioTrack
        PhysioTrack.process(config)
        
        # Rename output video to have the assessment ID
        output_files = list(Path(output_dir).glob("*_Sports2D.mp4"))
        if output_files:
            os.rename(output_files[0], Path(output_dir) / f"{assessment_id}.mp4")
        
        # Update status
        with open(status_file, "w") as f:
            json.dump({
                "status": "complete",
                "message": "Video processing complete"
            }, f)
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update status with error
        with open(status_file, "w") as f:
            json.dump({
                "status": "error",
                "message": f"Error processing video: {str(e)}"
            }, f)