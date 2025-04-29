#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import cv2
from pathlib import Path
import toml
from threading import Thread
from queue import Queue

# Load config for pose tracker
CONFIG_PATH = Path(__file__).parent / 'physiotrack' / 'Demo' / 'Config_demo.toml'
config_dict = toml.load(CONFIG_PATH)

app = FastAPI()

cap = None
pose_tracker = None
angle_dict = None
keypoints_ids = None
keypoints_names = None
model_root = None

@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
    <html><head><title>PhysioTrack Demo</title></head>
    <body>
      <h1>PhysioTrack Demo</h1>
      <ul>
        <li><a href="/webcam">Webcam Analysis</a></li>
        <li><a href="/upload">Upload Video</a></li>
      </ul>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/webcam", response_class=HTMLResponse)
async def webcam():
    global cap, pose_tracker, angle_dict, keypoints_ids, keypoints_names, model_root
    cap = cv2.VideoCapture(0)
    pose_tracker, angle_dict, keypoints_ids, keypoints_names, model_root = initialize_pose_tracker(config_dict)
    html = """
    <html><head><title>Webcam Analysis</title></head>
    <body>
      <h1>Webcam Analysis</h1>
      <img src="/video_feed" width="800" />
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/upload", response_class=HTMLResponse)
async def upload_form():
    html = """
    <html><head><title>Upload Video</title></head>
    <body>
      <h1>Upload Video</h1>
      <form action="/upload" enctype="multipart/form-data" method="post">
        <input type="file" name="file" accept="video/*" />
        <button type="submit">Upload</button>
      </form>
    </body></html>
    """
    return HTMLResponse(html)

@app.post("/upload", response_class=HTMLResponse)
async def upload_video(file: UploadFile = File(...)):
    global cap, pose_tracker, angle_dict, keypoints_ids, keypoints_names, model_root
    upload_dir = Path(__file__).parent / 'uploads'
    upload_dir.mkdir(exist_ok=True)
    filepath = upload_dir / file.filename
    with open(filepath, "wb") as f:
        f.write(await file.read())
    cap = cv2.VideoCapture(str(filepath))
    pose_tracker, angle_dict, keypoints_ids, keypoints_names, model_root = initialize_pose_tracker(config_dict)
    html = f"""
    <html><head><title>Video Uploaded</title></head>
    <body>
      <h1>File uploaded: {file.filename}</h1>
      <a href="/video_analysis">Start Analysis</a>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/video_analysis", response_class=HTMLResponse)
async def video_analysis():
    html = """
    <html><head><title>Video Analysis</title></head>
    <body>
      <h1>Video Analysis</h1>
      <img src="/video_feed" width="800" />
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/video_feed")
def video_feed():
    def gen():
        while cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame, pose_tracker, config_dict, angle_dict, keypoints_ids, keypoints_names, model_root)
            _, buffer = cv2.imencode('.jpg', processed)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        if cap is not None:
            cap.release()
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    uvicorn.run("main-demo:app", host="0.0.0.0", port=8000, reload=True)
