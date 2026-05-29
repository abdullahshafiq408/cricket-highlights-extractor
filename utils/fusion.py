"""
utils/fusion.py
───────────────
The fusion layer — the heart of the ML pipeline.

** CURRENT TEST LOGIC **
- NLP and Visual ML models are temporarily disabled (Weight = 0.0)
- Cricsheet is treated as ground truth.
- Audio Spikes rank the intensity of the event.
- Clip windows are heavily shifted backwards to account for "Scoreboard Lag" 
  (the delay between the action and the TV graphic updating).
"""

import os
import numpy as np
from models.ml_classifier import get_ml_score_at
from models.frame_classifier import get_visual_score_at
from utils.audio_utils import get_excitement_at
from utils.ocr_utils import resolve_timestamp


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT WEIGHTS — EXPERIMENTAL LOGIC
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "base_priority": 0.60,  # 60% - Guarantees the Cricsheet event makes the cut
    "audio_score":   0.40,  # 40% - Uses crowd roar to rank the best moments
    "nlp_score":     0.00,  # Disabled for this test
    "visual_score":  0.00,  # Disabled for this test
}


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ALL EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def score_events(
    events:              list[dict],     # from json_utils.extract_events()
    alignment_map:       dict,           # from ocr_utils.build_alignment_map()
    # from nlp_classifier.classify_all_segments()
    classified_segments: list[dict],
    # from audio_utils.detect_audio_spikes()
    audio_spikes:        list[dict],
    # from frame_classifier.score_all_frames()
    scored_frames:       list[dict],
    weights:             dict | None = None,
) -> list[dict]:
    """
    Compute a final highlight score. NLP and Visual scores are calculated 
    to prevent UI crashes, but do not affect the math.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Normalise weights to sum to 1.0 (safety check)
    total_w = sum(weights.values())
    w = {k: v / total_w for k, v in weights.items()}

    scored = []

    # --- CROSS-PLATFORM SMART PATHING ---
    # Automatically detect if we are running in the cloud (Colab) or locally (VSCode)
    frame_dir = '/content/frames' if os.path.exists(
        '/content/frames') else 'frames'

    for event in events:
        # Extract innings, default to 1 if it's missing for some reason
        innings = event.get("innings", 1)
        over = event["over"]
        ball = event["ball"]
        base_prio = event.get("base_priority", 0.5)

        # ── Step 1: resolve video timestamp ──────────────────────────────────
        # UPDATED: Now passing 'innings' to handle the dual-innings map
        timestamp = resolve_timestamp(innings, over, ball, alignment_map)
        if timestamp is None:
            # No alignment data — skip this event
            continue

        # ── Step 2: pull scores from each signal at this timestamp ───────────
        nlp_score = get_ml_score_at(classified_segments, timestamp)
        audio_score = get_excitement_at(audio_spikes,       timestamp)
        visual_score = get_visual_score_at(scored_frames,    timestamp)

        # ── Step 3: weighted fusion ───────────────────────────────────────────
        final_score = (
            w["base_priority"] * base_prio +
            w["nlp_score"] * nlp_score +
            w["audio_score"] * audio_score +
            w["visual_score"] * visual_score
        )

        # ── Step 4: clip window ───────────────────────────────────────────────
        pre_buffer = _get_pre_buffer(event["event_type"])
        post_buffer = _get_post_buffer(event["event_type"])

        scored.append({
            **event,
            "timestamp_sec": round(timestamp,    2),
            "nlp_score":     round(nlp_score,    4),
            "audio_score":   round(audio_score,  4),
            "visual_score":  round(visual_score, 4),
            "final_score":   round(final_score,  4),
            "clip_start":    max(0, timestamp - pre_buffer),
            "clip_end":      timestamp + post_buffer,
            # Added Telemetry Path
            "debug_ocr_frame": os.path.join(frame_dir, f"frame_{int(timestamp)}.jpg")
        })

    # Sort by final_score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    print(f"[fusion] Scored {len(scored)} events. "
          f"Top score: {scored[0]['final_score']:.3f}" if scored else
          "[fusion] No events scored.")
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# FILTER BY THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def filter_highlights(
    scored_events:  list[dict],
    threshold:      float = 0.40,
    max_clips:      int | None = None,  # <-- Set to None by default
) -> list[dict]:
    """
    Keep only events above the confidence threshold.
    If max_clips is None, it dynamically keeps ALL events that pass the threshold!
    Returns events in CHRONOLOGICAL ORDER.
    """
    # 1. Filter out the boring stuff
    filtered = [e for e in scored_events if e["final_score"] >= threshold]

    # 2. Only cap the list if a hard limit was explicitly asked for
    if max_clips is not None:
        filtered = filtered[:max_clips]

    # 3. Re-sort chronologically for the final reel
    filtered.sort(key=lambda x: x["timestamp_sec"])

    print(f"[fusion] {len(filtered)} events passed threshold {threshold} "
          f"(from {len(scored_events)} total)")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# CLIP BUFFER HELPERS (SCOREBOARD LAG ADJUSTED)
# ─────────────────────────────────────────────────────────────────────────────

def _get_pre_buffer(event_type: str) -> float:
    """
    How many seconds BEFORE the OCR timestamp to start the clip.
    Shifted heavily backwards to account for TV broadcast graphics delay.
    """
    buffers = {
        "wicket":     15.0,
        "boundary_6": 12.0,
        "boundary_4": 12.0,
    }
    return buffers.get(event_type, 6.0)


def _get_post_buffer(event_type: str) -> float:
    """
    How many seconds AFTER the OCR timestamp to end the clip.
    Shifted to be very short, as the action has largely concluded by the 
    time the graphic updates.
    """
    buffers = {
        "wicket":     6.0,
        "boundary_6": 4.0,
        "boundary_4": 4.0,
    }
    return buffers.get(event_type, 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING SUMMARY (for the Streamlit UI)
# ─────────────────────────────────────────────────────────────────────────────

def get_score_breakdown(event: dict) -> dict:
    """
    Return a human-readable breakdown of why an event was scored the way it was.
    Streamlit will still display all 4 bars, but NLP and Visual will be mathematically inert.
    """
    return {
        "Cricsheet priority": event.get("base_priority",  0.0),
        "NLP (commentary)":   event.get("nlp_score",      0.0),
        "Audio (crowd)":      event.get("audio_score",    0.0),
        "Visual (YOLOv8)":    event.get("visual_score",   0.0),
        "Final score":        event.get("final_score",    0.0),
    }
