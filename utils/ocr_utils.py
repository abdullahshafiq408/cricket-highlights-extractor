"""
utils/ocr_utils.py
──────────────────
Reads the scoreboard HUD (Heads-Up Display) overlay that appears in
broadcast cricket footage.

Why do we need this?
────────────────────
Cricsheet tells us WHAT happened (ball 14.3 = wicket) but NOT WHEN in the
video it happened (e.g. at 01:23:45).

By reading the scoreboard with OCR, we can build a mapping:
    over.ball  →  video_timestamp_seconds
"""

import re
import numpy as np
import cv2
import easyocr
from tqdm import tqdm

# Safe Streamlit import so the script works in both Colab and VSCode natively
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

_reader = None


def cache_wrapper(func):
    if HAS_STREAMLIT:
        return st.cache_resource(func)
    return func


@cache_wrapper
def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        print("[ocr_utils] Loading EasyOCR model (first run may take a minute)...")
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader

# ─────────────────────────────────────────────────────────────────────────────
# 1. CROP THE HUD REGION
# ─────────────────────────────────────────────────────────────────────────────


def crop_hud(frame: np.ndarray, bottom_fraction: float = 0.15, right_crop: float = 0.5) -> np.ndarray:
    """SPEED OPTIMIZED: Lighter 1.5x magnification + Otsu's smart thresholding."""
    h, w = frame.shape[:2]
    start_row = int(h * (1 - bottom_fraction))
    end_col = int(w * (1 - right_crop))
    roi = frame[start_row:h, 0:end_col]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=1.5, fy=1.5,
                          interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(
        enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACT TEXT
# ─────────────────────────────────────────────────────────────────────────────


def read_frame_text(processed_frame: np.ndarray) -> str:
    reader = _get_reader()
    results = reader.readtext(processed_frame, detail=0, paragraph=True)
    return " ".join(results)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PARSE OVER.BALL FROM TEXT
# ─────────────────────────────────────────────────────────────────────────────


def parse_over_ball(text: str) -> tuple[int, int] | None:
    cleaned = text.replace("I", "1").replace(
        "l", "1").replace("O", "0").replace("o", "0")
    score_pattern = re.compile(r"\b\d{1,3}\s?[-/]\s?\d{1,2}\b")
    marked_text = score_pattern.sub("[SCORE]", cleaned)

    match_A = re.findall(r"\b(\d{1,2})\s?\.\s?([0-6])\b", marked_text)
    if match_A:
        over, ball = int(match_A[-1][0]), int(match_A[-1][1])
        if 0 <= over <= 50:
            return (over, ball)

    match_B = re.findall(r"\b(\d{1,3})\s?[/]?\s?(?:20|50)\b", marked_text)
    if match_B:
        num_str = match_B[-1]
        if len(num_str) == 3 or (num_str.startswith("0") and len(num_str) > 1):
            over, ball = int(num_str[:-1]), int(num_str[-1])
        else:
            over, ball = int(num_str), 0
        if 0 <= over <= 50 and 0 <= ball <= 6:
            return (over, ball)

    match_C = re.findall(r"\b(\d{1,2})\s?\.\b", marked_text)
    if match_C:
        over = int(match_C[-1])
        if 0 <= over <= 50:
            return (over, 0)

    match_D = re.search(r"\[SCORE\][^\d]*(\d{1,2})", marked_text)
    if match_D:
        over = int(match_D.group(1))
        if 0 <= over <= 50:
            return (over, 0)

    return None

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD THE TEMPORAL ALIGNMENT MAP (DUAL INNINGS TRACKER)
# ─────────────────────────────────────────────────────────────────────────────


def build_alignment_map(frames: list[dict]) -> dict[str, float]:
    print(f"[ocr_utils] Scanning {len(frames)} frames for scoreboard text...")

    raw_detections = []

    # 1. Read all frames chronologically
    for frame_info in tqdm(frames, desc="OCR Scanning"):
        img = cv2.imread(frame_info["frame_path"])
        if img is None:
            continue

        hud = crop_hud(img)
        text = read_frame_text(hud)
        result = parse_over_ball(text)

        if result:
            over, ball = result
            raw_detections.append({
                "timestamp": frame_info["timestamp_sec"],
                "over_val": float(f"{over}.{ball}"),
                "over": over,
                "ball": ball
            })

    # 2. ── BULLETPROOF DUAL-INNINGS SANITIZER ──
    clean_map = {}
    current_innings = 1
    highest_over = -1.0
    last_valid_ts = 0.0

    for i, det in enumerate(raw_detections):
        val = det["over_val"]
        ts = det["timestamp"]

        # --- B. INNINGS SWAP LOCK (PROGRESSION DETECTOR) ---
        if current_innings == 1 and highest_over > 4.0 and val <= 0.5:
            unique_future_lows = set()
            for future_det in raw_detections[i+1: i+21]:
                f_val = future_det["over_val"]
                if f_val <= 2.0:
                    unique_future_lows.add(f_val)

            if len(unique_future_lows) >= 2:
                current_innings = 2
                highest_over = -1.0
                print(
                    f"\n[ocr_utils] 🔄 Detected 2nd Innings via Timeline Progression around {ts/60:.1f} minutes!")
            else:
                continue

        key = f"{current_innings}_{det['over']}.{det['ball']}"
        last_seen_key = key + "_last"  # HIDDEN METADATA FOR TRAILING EDGE

        # --- A. INITIALIZATION LOCK ---
        if highest_over == -1.0:
            if val <= 5.0:
                if key not in clean_map:
                    clean_map[key] = ts
                # Constantly update the trailing edge
                clean_map[last_seen_key] = ts
                highest_over = val
                last_valid_ts = ts
            continue

        # --- C. TIME-AWARE RATCHET (BASE-6 PHYSICS LOCK) ---
        if highest_over != -1.0:
            delta_overs = val - highest_over
            delta_time = ts - last_valid_ts

            if delta_overs < 0:
                continue

            val_over = int(val)
            val_ball = round((val - val_over) * 10)
            val_total_balls = (val_over * 6) + val_ball

            high_over = int(highest_over)
            high_ball = round((highest_over - high_over) * 10)
            high_total_balls = (high_over * 6) + high_ball

            delta_balls = val_total_balls - high_total_balls
            max_possible_balls = (delta_time / 25.0) + 1

            if delta_balls > max_possible_balls:
                continue

            if key not in clean_map:
                clean_map[key] = ts
            # Constantly update the trailing edge
            clean_map[last_seen_key] = ts

            highest_over = val
            last_valid_ts = ts

    # Filter out our hidden metadata keys for the console printout to keep it clean
    actual_anchors = [k for k in clean_map.keys() if not k.endswith("_last")]
    print(f"\n[ocr_utils] RAW DETECTIONS: {len(raw_detections)}")
    print(
        f"[ocr_utils] CLEANED ANCHORS ({len(actual_anchors)}): {actual_anchors}\n")

    # --- CROSS-PLATFORM TELEMETRY EXPORT ---
    import os
    import json

    # Smart Pathing: Cloud vs Local
    out_dir = '/content/drive/MyDrive' if os.path.exists(
        '/content/drive/MyDrive') else 'assets'
    os.makedirs(out_dir, exist_ok=True)

    try:
        with open(os.path.join(out_dir, 'ocr_telemetry.json'), 'w') as f:
            json.dump(raw_detections, f, indent=4)
        print(
            f"[ocr_utils] 💾 Saved OCR telemetry to {out_dir}/ocr_telemetry.json")
    except Exception as e:
        print(f"[ocr_utils] ⚠️ Could not save telemetry: {e}")

    return clean_map

# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER: resolve a Cricsheet event to a video timestamp
# ─────────────────────────────────────────────────────────────────────────────


def resolve_timestamp(
    innings: int,
    over: int,
    ball: int,
    alignment_map: dict[str, float],
    pre_buffer: float = 2.0,
) -> float | None:
    """
    STRICT TRANSITION MATCHING (TRAILING EDGE):
    """
    key = f"{innings}_{over}.{ball}"

    # --- THE CRICSHEET TO TV GRAPHIC TRANSLATION ---
    # If Cricsheet asks for the 6th ball (e.g., 4.6), translate it to the
    # TV graphic for the end of the over (e.g., 5.0) which you already have.
    if ball == 6:
        tv_key = f"{innings}_{over+1}.0"
        if tv_key in alignment_map:
            print(
                f"[ocr_utils] 🔗 Direct Map: Translated Cricsheet {over}.6 -> OCR {over+1}.0")
            return alignment_map[tv_key]

    # 1. If we have the exact ball, just return its normal starting timestamp
    if key in alignment_map:
        return alignment_map[key]

    # 2. Determine exact predecessors to look for.
    predecessors = []

    if ball > 1 and ball < 6:
        # For 14.3, look for the trailing edge of 14.2
        predecessors.append(f"{innings}_{over}.{ball-1}")
    elif ball == 1:
        # For 15.1, the TV likely showed 15.0 or 14.5 right before it
        predecessors.append(f"{innings}_{over}.0")
        predecessors.append(f"{innings}_{over-1}.5")
    elif ball == 0:
        predecessors.append(f"{innings}_{over-1}.5")

    # 3. Look for the Trailing Edge of the previous ball
    for prev_key in predecessors:
        last_seen_key = prev_key + "_last"
        if last_seen_key in alignment_map:
            transition_timestamp = alignment_map[last_seen_key]
            estimated_start = max(0.0, transition_timestamp - pre_buffer)
            print(
                f"[ocr_utils] 🎯 Strict Interpolation: {key} anchored to the end of {prev_key}")
            return estimated_start

    # 4. If the immediate predecessor is missing, abort.
    print(f"[ocr_utils] 🚫 Dropping {key} - Immediate anchor missing.")
    return None
# ─────────────────────────────────────────────────────────────────────────────
# 6. HELPER: Filter events to Match Video Duration
# ─────────────────────────────────────────────────────────────────────────────


def filter_valid_events(events: list[dict], alignment_map: dict[str, float]) -> list[dict]:
    """
    Passes events through the new resolve_timestamp logic so interpolated 
    events are correctly preserved and not accidentally dropped.
    """
    if not alignment_map:
        return events

    filtered = []

    for e in events:
        event_innings = e.get("innings", 1)

        # If resolve_timestamp can find it (or strictly interpolate it), keep it!
        ts = resolve_timestamp(
            event_innings, e['over'], e['ball'], alignment_map)

        if ts is not None:
            filtered.append(e)

    return filtered
