from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import uuid
import os
from pathlib import Path

from app.routers import assessment, exercise, utils

app = FastAPI(
    title="PhysioTrack API",
    description="API for physiotherapy assessment using computer vision",
    version="0.1.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(assessment.router, prefix="/api/v1/assessment", tags=["Assessment"])
app.include_router(exercise.router, prefix="/api/v1/exercise", tags=["Exercise"])
app.include_router(utils.router, prefix="/api/v1/utils", tags=["Utilities"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to PhysioTrack API", "version": "0.1.0"}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)