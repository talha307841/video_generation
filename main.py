from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from mochi import MochiModel  # Assuming Mochi-1 has a Python interface
import os
import uuid
from fastapi.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Define directories
VIDEO_OUTPUT_DIR = "generated_videos"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# Initialize Mochi-1 model
def initialize_model():
    # Load Mochi-1 with GPU acceleration if available
    model = MochiModel(device="cuda" if torch.cuda.is_available() else "cpu")
    return model

model = initialize_model()

# Define request model
class VideoRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 5  # default duration of video in seconds

# ThreadPoolExecutor for handling concurrency
executor = ThreadPoolExecutor(max_workers=4)

# Video generation function
def generate_video(prompt: str, duration: int, output_path: str):
    try:
        # Use Mochi-1 for video generation
        video_content = model.generate(prompt=prompt, duration=duration)
        with open(output_path, "wb") as f:
            f.write(video_content)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Video generation failed: {str(e)}")

@app.post("/generate_video/")
def create_video(request: VideoRequest, background_tasks: BackgroundTasks):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    video_id = str(uuid.uuid4())
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")

    # Schedule video generation in the background
    background_tasks.add_task(generate_video, request.prompt, request.duration, output_path)

    return {"status": "Processing", "video_id": video_id, "download_url": f"/get_video/{video_id}"}

@app.get("/get_video/{video_id}")
def get_video(video_id: str):
    video_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path)

@app.get("/")
def read_root():
    return {"message": "Text-to-Video API is running."}



