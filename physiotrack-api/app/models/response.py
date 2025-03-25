# models/response.py
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

class AssessmentResponse(BaseModel):
    assessment_id: str
    status: str
    message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    video_url: Optional[str] = None