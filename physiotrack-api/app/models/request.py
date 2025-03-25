# models/request.py
from pydantic import BaseModel
from typing import List, Optional

class ROMAssessmentParams(BaseModel):
    height: Optional[float] = 1.7
    visible_side: Optional[str] = "auto"
    time_range: Optional[List[float]] = None
    joint_angles: Optional[List[str]] = None
    segment_angles: Optional[List[str]] = None

class ExerciseGuidanceParams(BaseModel):
    exercise_type: str
    target_reps: Optional[int] = 10
    height: Optional[float] = 1.7