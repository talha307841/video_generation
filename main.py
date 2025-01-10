from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
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
    model = MochiPipeline.from_pretrained("genmo/mochi-1-preview")
    model.enable_model_cpu_offload()
    model.enable_vae_tiling()
    return model

model = initialize_model()

# Define request model
class VideoRequest(BaseModel):
    prompt: str
    num_frames: Optional[int] = 84  # default number of frames
    fps: Optional[int] = 30        # default frames per second

# ThreadPoolExecutor for handling concurrency
executor = ThreadPoolExecutor(max_workers=4)

# Video generation function
def generate_video(prompt: str, num_frames: int, fps: int, output_path: str):
    try:
        # Use Mochi-1 for video generation
        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            frames = model(prompt, num_frames=num_frames).frames[0]
        export_to_video(frames, output_path, fps=fps)
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
    background_tasks.add_task(generate_video, request.prompt, request.num_frames, request.fps, output_path)

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
