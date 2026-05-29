"""
models/frame_classifier.py
──────────────────────────
Uses YOLOv8 to detect visual events in extracted video frames.

Why YOLOv8?
───────────
YOLO (You Only Look Once) is a family of real-time object detection models.
YOLOv8 by Ultralytics is the latest version — fast, accurate, easy to use.

For a beginner project we use it in two ways:
1. OBJECT DETECTION (default) — detect objects like "person", "sports ball"
   in each frame. Certain patterns (many players running together, ball
   near the stumps) correlate with exciting events.
2. FRAME-LEVEL CLASSIFICATION — we compute a simple "action score" from
   the detected objects and their positions.

In a more advanced version, you would:
- Fine-tune YOLOv8 on a cricket-specific dataset with labels like
  "boundary_rope", "stumps_fallen", "fielder_diving", etc.
- But for a beginner project the pre-trained COCO model + heuristics works.

Cricket-relevant COCO classes we care about:
    person  (class 0)  → fielders, batsmen, umpires
    sports ball (class 32) → the cricket ball (sometimes detected)
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# YOLO MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

_model = None

def _get_model(model_name: str = "yolov8n.pt") -> YOLO:
    """
    Load YOLOv8 model (cached).

    model_name options:
        yolov8n.pt  → nano, fastest (~3MB)   ← DEFAULT for beginner project
        yolov8s.pt  → small, more accurate
        yolov8m.pt  → medium
        yolov8l.pt  → large, most accurate but slow
    """
    global _model
    if _model is None:
        print(f"[frame_classifier] Loading {model_name}...")
        _model = YOLO(model_name)   # auto-downloads on first run
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# SCORE A SINGLE FRAME
# ─────────────────────────────────────────────────────────────────────────────

# COCO class IDs we care about
PERSON_CLASS_ID      = 0
SPORTS_BALL_CLASS_ID = 32

def score_frame(frame_path: str) -> dict:
    """
    Run YOLOv8 on a single frame and return a visual excitement score.

    The score is based on:
    - How many people are visible (more = more crowded/active play)
    - Whether any "sports ball" is detected
    - The average confidence of person detections (higher = clearer action)
    - Spatial clustering: people bunched together suggests a wicket celebration
      or a fielder running after a boundary

    Returns:
    {
        "frame_path":        "...",
        "n_persons":         8,
        "ball_detected":     True,
        "visual_score":      0.74,
        "detections":        [...raw YOLO results...]
    }
    """
    model = _get_model()

    img = cv2.imread(frame_path)
    if img is None:
        return {
            "frame_path":    frame_path,
            "n_persons":     0,
            "ball_detected": False,
            "visual_score":  0.0,
            "detections":    [],
        }

    results = model(img, verbose=False)[0]  # [0] because we pass one image

    # Parse detections
    persons   = []
    has_ball  = False
    all_boxes = results.boxes

    for box in all_boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy       = box.xyxy[0].tolist()   # [x1, y1, x2, y2]

        if cls_id == PERSON_CLASS_ID and confidence > 0.4:
            persons.append({"conf": confidence, "box": xyxy})
        elif cls_id == SPORTS_BALL_CLASS_ID and confidence > 0.3:
            has_ball = True

    # ── Heuristic scoring ────────────────────────────────────────────────────

    # 1. More visible people → more action (capped, normalised to 0–1)
    person_score = min(len(persons) / 12.0, 1.0)

    # 2. Ball detected → likely delivery or hit in progress
    ball_score = 0.4 if has_ball else 0.0

    # 3. Spatial clustering of persons
    #    If people are bunched into a small area → celebration / run-out chase
    clustering_score = 0.0
    if len(persons) >= 3:
        centres_x = [(b["box"][0] + b["box"][2]) / 2 for b in persons]
        centres_y = [(b["box"][1] + b["box"][3]) / 2 for b in persons]
        spread_x  = np.std(centres_x)
        spread_y  = np.std(centres_y)
        # Low spread = clustered = exciting; high spread = spread out = boring
        clustering_score = max(0.0, 1.0 - (spread_x + spread_y) / 500.0)

    # Weighted combination
    visual_score = (
        0.40 * person_score +
        0.35 * ball_score +
        0.25 * clustering_score
    )

    return {
        "frame_path":    frame_path,
        "n_persons":     len(persons),
        "ball_detected": has_ball,
        "clustering_score": clustering_score,
        "visual_score":  round(visual_score, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ALL FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def score_all_frames(frames: list[dict]) -> list[dict]:
    """
    Run the visual scorer over all extracted frames.

    frames : output of video_utils.extract_frames()
             [{"frame_path": ..., "timestamp_sec": ...}, ...]

    Returns the same list with added scoring fields:
    [
      {
        "frame_path":    "...",
        "timestamp_sec": 42.0,
        "n_persons":     7,
        "ball_detected": False,
        "visual_score":  0.55,
      },
      ...
    ]
    """
    print(f"[frame_classifier] Scoring {len(frames)} frames with YOLOv8...")
    results = []

    for frame_info in tqdm(frames, desc="Visual scoring"):
        scores = score_frame(frame_info["frame_path"])
        results.append({
            **frame_info,
            **scores,
        })

    avg_score = np.mean([r["visual_score"] for r in results])
    print(f"[frame_classifier] Average visual score: {avg_score:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: get visual score at a timestamp
# ─────────────────────────────────────────────────────────────────────────────

def get_visual_score_at(
    scored_frames: list[dict],
    timestamp_sec: float,
    window_sec:    float = 4.0,
) -> float:
    """
    Get the peak visual score within ±window_sec of a timestamp.

    Used by the fusion layer.
    """
    relevant = [
        f["visual_score"]
        for f in scored_frames
        if abs(f["timestamp_sec"] - timestamp_sec) <= window_sec
    ]
    return max(relevant) if relevant else 0.0
