import os
import re
import json
import math
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import yt_dlp
from celery import Task
from celery_app import celery_app

CLIPS_DIR = Path("./clips")
CLIPS_DIR.mkdir(exist_ok=True)

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


def update_progress(task, progress: int, message: str, stage: str = ""):
    task.update_state(
        state="PROGRESS",
        meta={"progress": progress, "message": message, "stage": stage},
    )


def run_cmd(cmd: List[str], check=True, capture=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def get_video_duration(path: str) -> float:
    result = run_cmd([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        path,
    ])
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def detect_scenes(video_path: str, threshold: float = 0.3) -> List[float]:
    """Use ffmpeg scene detection to find scene changes."""
    result = run_cmd([
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ], capture=True, check=False)

    timestamps = []
    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            match = re.search(r"pts_time:([\d.]+)", line)
            if match:
                timestamps.append(float(match.group(1)))
    return sorted(timestamps)


def score_segments(duration: float, scene_times: List[float], num_clips: int) -> List[Tuple[float, float]]:
    """Select best segments for clips based on scene density and spacing."""
    clip_length = 45  # target seconds per clip

    if not scene_times:
        # No scenes detected — evenly distribute
        step = duration / num_clips
        return [(i * step, min(i * step + clip_length, duration)) for i in range(num_clips)]

    # Score each scene by proximity to others (density = interesting)
    scored = []
    for i, t in enumerate(scene_times):
        nearby = sum(1 for s in scene_times if abs(s - t) < 30)
        scored.append((nearby, t))

    scored.sort(reverse=True)

    # Select non-overlapping segments
    selected = []
    for _, center in scored:
        start = max(0, center - clip_length / 2)
        end = min(duration, start + clip_length)
        start = max(0, end - clip_length)

        # Check overlap
        overlap = any(
            not (end <= s or start >= e)
            for s, e in selected
        )
        if not overlap:
            selected.append((start, end))

        if len(selected) >= num_clips:
            break

    # Fill remaining slots evenly if needed
    if len(selected) < num_clips:
        step = duration / (num_clips + 1)
        for i in range(1, num_clips + 1):
            center = i * step
            start = max(0, center - clip_length / 2)
            end = min(duration, start + clip_length)
            overlap = any(not (end <= s or start >= e) for s, e in selected)
            if not overlap:
                selected.append((start, end))
            if len(selected) >= num_clips:
                break

    return sorted(selected, key=lambda x: x[0])[:num_clips]


def extract_clip(
    video_path: str,
    output_path: str,
    start: float,
    end: float,
    index: int,
    task,
    base_progress: int,
) -> bool:
    """Extract, convert to vertical 9:16, add captions via ffmpeg."""
    duration = end - start

    # Build FFmpeg filter for vertical crop + zoom + captions
    vf = (
        "scale=iw*4:ih*4,"          # upscale first
        "crop=ih*9/16:ih,"           # crop to 9:16
        "scale=1080:1920,"           # final resolution
        "zoompan=z='min(zoom+0.001,1.05)':d=25:s=1080x1920,"  # subtle zoom
        "unsharp=5:5:1.0:5:5:0.0"   # sharpness
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def add_captions_to_clip(video_path: str, output_path: str, whisper_available: bool) -> bool:
    """Add burned-in captions using Whisper if available, else skip."""
    if not whisper_available:
        shutil.copy(video_path, output_path)
        return True

    srt_path = video_path.replace(".mp4", ".srt")

    # Run Whisper
    result = subprocess.run(
        ["whisper", video_path, "--model", "tiny", "--output_format", "srt",
         "--output_dir", str(Path(video_path).parent), "--language", "en"],
        capture_output=True, text=True,
    )

    if result.returncode != 0 or not Path(srt_path).exists():
        shutil.copy(video_path, output_path)
        return True

    # Burn captions in
    style = (
        "FontName=Arial,FontSize=18,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
        "Bold=1,Outline=2,Shadow=1,Alignment=2,MarginV=80"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}:force_style='{style}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        output_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        shutil.copy(video_path, output_path)
    return True


def check_whisper():
    try:
        result = subprocess.run(["whisper", "--help"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


@celery_app.task(bind=True, name="tasks.process_video")
def process_video_task(self, url: str, num_clips: int, job_id: str):
    job_dir = CLIPS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    whisper_ok = check_whisper()

    try:
        # ── Stage 1: Fetch video info ──────────────────────────────────────
        update_progress(self, 5, "Fetching video information...", "analyzing")

        ydl_info_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        title = info.get("title", "Unknown")
        thumbnail = info.get("thumbnail", "")
        thumbnails = info.get("thumbnails", [])
        if thumbnails:
            hq = [t for t in thumbnails if t.get("width", 0) >= 640]
            thumbnail = hq[-1]["url"] if hq else thumbnails[-1]["url"]
        duration = float(info.get("duration", 0))

        if duration > 3600:
            raise ValueError("Video too long (max 60 minutes)")

        # ── Stage 2: Download video ────────────────────────────────────────
        update_progress(self, 10, "Downloading video...", "downloading")

        raw_path = str(job_dir / "raw.mp4")
        ydl_dl_opts = {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "outtmpl": raw_path,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(ydl_dl_opts) as ydl:
            ydl.download([url])

        if not Path(raw_path).exists():
            # Try alternate filename (yt-dlp may add extension)
            candidates = list(job_dir.glob("raw.*"))
            if candidates:
                raw_path = str(candidates[0])
            else:
                raise FileNotFoundError("Video download failed")

        update_progress(self, 30, "Analyzing video content...", "analyzing")

        # ── Stage 3: Scene detection ───────────────────────────────────────
        actual_duration = get_video_duration(raw_path)
        scene_times = detect_scenes(raw_path)

        update_progress(self, 45, f"Found {len(scene_times)} scene changes, selecting best moments...", "analyzing")

        # ── Stage 4: Select segments ───────────────────────────────────────
        segments = score_segments(actual_duration, scene_times, num_clips)

        # ── Stage 5: Extract clips ─────────────────────────────────────────
        clips_info = []
        for i, (start, end) in enumerate(segments):
            prog = 50 + int((i / len(segments)) * 45)
            update_progress(
                self,
                prog,
                f"Rendering clip {i + 1} of {len(segments)}...",
                "rendering",
            )

            clip_raw = str(job_dir / f"clip_{i+1}_raw.mp4")
            clip_final = str(job_dir / f"clip_{i+1}.mp4")

            ok = extract_clip(raw_path, clip_raw, start, end, i, self, prog)
            if not ok:
                continue

            add_captions_to_clip(clip_raw, clip_final, whisper_ok)

            # Clean up intermediate
            if Path(clip_raw).exists() and clip_raw != clip_final:
                os.remove(clip_raw)

            file_size = Path(clip_final).stat().st_size if Path(clip_final).exists() else 0
            clip_duration = round(end - start, 1)

            clips_info.append({
                "id": f"{job_id}_{i+1}",
                "index": i + 1,
                "filename": f"clip_{i+1}.mp4",
                "url": f"{BASE_URL}/clips/{job_id}/clip_{i+1}.mp4",
                "start": round(start, 1),
                "end": round(end, 1),
                "duration": clip_duration,
                "size_mb": round(file_size / 1024 / 1024, 1),
                "label": f"Clip {i+1}",
            })

        # Clean up raw download
        if Path(raw_path).exists():
            os.remove(raw_path)

        update_progress(self, 99, "Finalizing...", "rendering")

        return {
            "clips": clips_info,
            "title": title,
            "thumbnail": thumbnail,
            "url": url,
            "num_clips": len(clips_info),
        }

    except Exception as e:
        # Clean up on failure
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
