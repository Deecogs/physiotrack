from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
import os
from pathlib import Path
import json
import tempfile

from app.services.pose_service import process_video
from app.services.analysis_service import analyze_rom
from app.models.request import ROMAssessmentParams
from app.models.response import AssessmentResponse

router = APIRouter()

@router.post("/rom", response_model=AssessmentResponse)
async def assess_rom(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    params: Optional[str] = Form("{}")
):
    """
    Analyze range of motion from a video
    
    - **video**: Video file to analyze
    - **params**: JSON string of assessment parameters
    """
    # Create a temporary file to store the uploaded video
    temp_dir = Path("static/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique ID for this assessment
    assessment_id = str(uuid.uuid4())
    
    # Create a directory for this assessment
    assessment_dir = temp_dir / assessment_id
    assessment_dir.mkdir(exist_ok=True)
    
    # Save the uploaded video
    video_path = assessment_dir / f"input{Path(video.filename).suffix}"
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    # Parse parameters
    try:
        params_dict = json.loads(params)
        assessment_params = ROMAssessmentParams(**params_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in params")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
    
    # Process the video using PhysioTrack (in background)
    background_tasks.add_task(
        process_video,
        str(video_path),
        str(assessment_dir),
        assessment_params
    )
    
    # Return response with assessment ID
    return {
        "assessment_id": assessment_id,
        "status": "processing",
        "message": "Video uploaded and being processed. Check status endpoint for results."
    }

@router.get("/rom/{assessment_id}", response_model=AssessmentResponse)
async def get_rom_assessment(assessment_id: str):
    """
    Get the results of a range of motion assessment
    
    - **assessment_id**: ID of the assessment to retrieve
    """
    assessment_dir = Path(f"static/temp/{assessment_id}")
    if not assessment_dir.exists():
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    # Check if processing is complete
    status_file = assessment_dir / "status.json"
    if not status_file.exists():
        return {
            "assessment_id": assessment_id,
            "status": "processing",
            "message": "Assessment is still being processed"
        }
    
    # Read results
    with open(status_file, "r") as f:
        status = json.load(f)
    
    # If processing complete, analyze ROM from angles file
    if status["status"] == "complete":
        angles_file = assessment_dir / f"{assessment_id}_angles_person00.mot"
        if angles_file.exists():
            rom_analysis = analyze_rom(angles_file)
            return {
                "assessment_id": assessment_id,
                "status": "complete",
                "results": rom_analysis,
                "video_url": f"/static/temp/{assessment_id}/{assessment_id}.mp4"
            }
    
    # Return current status
    return {
        "assessment_id": assessment_id,
        "status": status["status"],
        "message": status.get("message", "")
    }