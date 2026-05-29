"""
utils/video_utils.py
────────────────────
Everything related to reading a video file:
  - extract_frames()   → saves frames as JPEGs at a chosen rate
  - cut_clip()         → cuts a short clip around a timestamp
  - stitch_clips()     → joins a list of clips into one final highlight reel

We use OpenCV for frame extraction (fast, no re-encoding)
and MoviePy for clip cutting / stitching (high-level, easy API).
"""

import os
import cv2
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
)
from tqdm import tqdm              

def extract_frames(video_path: str, output_dir: str, fps: int = 2) -> list[dict]:
    """
    Extract frames from a video at `fps` frames per second.

    Why do we extract frames?
    → We need still images to run OCR (reading the scoreboard) and
      the visual event detector (YOLOv8) on. Doing this up-front is
      faster than decoding the video repeatedly.

    Parameters
    ----------
    video_path  : path to the input MP4/MKV
    output_dir  : folder where JPEG frames will be saved
    fps         : how many frames to extract per second of video
                  (2 is a good balance of speed vs. detail)

    Returns
    -------
    A list of dicts: [{"frame_path": "...", "timestamp_sec": 12.5}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)          # open the video file
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    frame_interval = max(1, int(video_fps / fps))

    saved = []
    frame_idx = 0

    print(
        f"[video_utils] Video: {duration_sec:.1f}s  |  Extracting at {fps} fps...")

    with tqdm(total=int(duration_sec * fps), desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()        
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / video_fps
                filename = f"frame_{frame_idx:07d}.jpg"
                filepath = os.path.join(output_dir, filename)

                cv2.imwrite(filepath, frame)  

                saved.append({
                    "frame_path":    filepath,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "frame_idx":     frame_idx,
                })
                pbar.update(1)

            frame_idx += 1

    cap.release()
    print(f"[video_utils] Saved {len(saved)} frames to {output_dir}")
    return saved

def cut_clip(video_path: str, start_time: float, end_time: float, out_path: str) -> str | None:
    import subprocess
    import imageio_ffmpeg  # MoviePy automatically installed this on your PC!
    
    try:
        if start_time is None or end_time is None:
            return None
            
        start = max(0.0, float(start_time))
        end = float(end_time)
        duration = end - start
        
     
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe,
            "-y",                   # Overwrite if file already exists
            "-ss", str(start),      # Seek exactly to our clip start
            "-i", video_path,       # Load the main video
            "-t", str(duration),    # Cut exactly this many seconds
            "-c:v", "libx264",      # Re-encode video to standard MP4
            "-c:a", "aac",          # Re-encode audio to standard MP4
            "-preset", "fast",      # Process it quickly
            out_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return out_path
    except Exception as e:
        print(f"[video_utils] ⚠ Failed to FFMPEG cut clip {out_path}: {e}")
        return None


def stitch_clips(clip_paths: list[str], out_path: str) -> bool:
    import os
    import subprocess
    import imageio_ffmpeg
    
    if not clip_paths:
        return False
        
    try:

        list_file_path = os.path.join(os.path.dirname(out_path), "concat_list.txt")
        
        with open(list_file_path, "w", encoding="utf-8") as f:
            for c in clip_paths:
                # FFMPEG prefers forward slashes for paths, even on Windows
                safe_path = os.path.abspath(c).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
   
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe,
            "-y",                   # Overwrite if file already exists
            "-f", "concat",         # Use the concatenation demuxer
            "-safe", "0",           # Allow absolute file paths
            "-i", list_file_path,   # Pass in our text file of clips
            "-c", "copy",           # STREAM COPY: Do not re-encode! (Lightning fast)
            out_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(list_file_path):
            os.remove(list_file_path)
            
        return True
        
    except Exception as e:
        print(f"[video_utils] ⚠ Failed to FFMPEG stitch clips: {e}")
        return False
