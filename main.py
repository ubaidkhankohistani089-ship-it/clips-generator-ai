import os
import uuid
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yt_dlp

from celery_app import celery_app
from tasks import process_video_task

app = FastAPI(title="ClipForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIPS_DIR = Path("./clips")
CLIPS_DIR.mkdir(exist_ok=True)
app.mount("/clips", StaticFiles(directory="clips"), name="clips")

PROJECTS_FILE = Path("./projects.json")


def load_projects():
    if PROJECTS_FILE.exists():
        return json.loads(PROJECTS_FILE.read_text())
    return {}


def save_projects(projects):
    PROJECTS_FILE.write_text(json.dumps(projects, indent=2))


class VideoInfoRequest(BaseModel):
    url: str


class GenerateClipsRequest(BaseModel):
    url: str
    num_clips: int = 5


def is_valid_youtube_url(url: str) -> bool:
    patterns = [
        r"^https?://(www\.)?youtube\.com/watch\?v=[\w-]+",
        r"^https?://youtu\.be/[\w-]+",
        r"^https?://(www\.)?youtube\.com/shorts/[\w-]+",
    ]
    return any(re.match(p, url) for p in patterns)


@app.get("/")
def root():
    return {"status": "ClipForge API running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/fetch-video-info")
def fetch_video_info(req: VideoInfoRequest):
    if not is_valid_youtube_url(req.url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 15,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=False)

        duration = info.get("duration", 0)
        if duration > 3600:
            raise HTTPException(status_code=400, detail="Video too long (max 60 min)")

        thumbnail = info.get("thumbnail", "")
        thumbnails = info.get("thumbnails", [])
        if thumbnails:
            hq = [t for t in thumbnails if t.get("width", 0) >= 640]
            thumbnail = hq[-1]["url"] if hq else thumbnails[-1]["url"]

        return {
            "title": info.get("title", "Unknown"),
            "thumbnail": thumbnail,
            "duration": duration,
            "channel": info.get("uploader", ""),
            "view_count": info.get("view_count", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        err = str(e)
        if "Private video" in err or "Sign in" in err:
            raise HTTPException(status_code=403, detail="Video is private or restricted")
        if "not available" in err.lower():
            raise HTTPException(status_code=404, detail="Video not available in your region")
        raise HTTPException(status_code=500, detail=f"Failed to fetch video info: {err}")


@app.post("/api/generate-clips")
def generate_clips(req: GenerateClipsRequest):
    if not is_valid_youtube_url(req.url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    num_clips = max(1, min(20, req.num_clips))
    job_id = str(uuid.uuid4())

    task = process_video_task.apply_async(
        args=[req.url, num_clips, job_id],
        task_id=job_id,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        return {"job_id": job_id, "status": "queued", "progress": 0, "message": "Waiting in queue..."}
    elif result.state == "PROGRESS":
        meta = result.info or {}
        return {
            "job_id": job_id,
            "status": "processing",
            "progress": meta.get("progress", 0),
            "message": meta.get("message", "Processing..."),
            "stage": meta.get("stage", ""),
        }
    elif result.state == "SUCCESS":
        info = result.info or {}
        clips = info.get("clips", [])
        # Save project
        projects = load_projects()
        if job_id not in projects:
            projects[job_id] = {
                "job_id": job_id,
                "title": info.get("title", "Unknown"),
                "thumbnail": info.get("thumbnail", ""),
                "clips": clips,
                "created_at": datetime.utcnow().isoformat(),
                "url": info.get("url", ""),
            }
            save_projects(projects)
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "message": "Ready to download!",
            "clips": clips,
            "title": info.get("title", ""),
            "thumbnail": info.get("thumbnail", ""),
        }
    elif result.state == "FAILURE":
        return {
            "job_id": job_id,
            "status": "failed",
            "progress": 0,
            "message": str(result.info),
        }
    else:
        return {"job_id": job_id, "status": result.state.lower(), "progress": 0}


@app.get("/api/clips/{job_id}")
def get_clips(job_id: str):
    result = celery_app.AsyncResult(job_id)
    if result.state != "SUCCESS":
        raise HTTPException(status_code=404, detail="Clips not ready yet")
    info = result.info or {}
    return {"clips": info.get("clips", []), "title": info.get("title", "")}


@app.get("/api/projects")
def get_projects():
    projects = load_projects()
    return {"projects": list(projects.values())}


@app.delete("/api/projects/{job_id}")
def delete_project(job_id: str):
    projects = load_projects()
    if job_id in projects:
        # Clean up clip files
        job_dir = CLIPS_DIR / job_id
        if job_dir.exists():
            import shutil
            shutil.rmtree(job_dir)
        del projects[job_id]
        save_projects(projects)
    return {"status": "deleted"}
