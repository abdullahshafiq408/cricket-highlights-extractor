"""
utils/audio_utils.py
────────────────────
Two separate jobs:

1. transcribe_audio()
   → Uses OpenAI Whisper to convert commentary speech → timestamped text.
   → Returns a list of segments: [{start, end, text}, ...]

2. detect_audio_spikes()
   → Uses librosa to measure the RMS (Root Mean Square) energy of the audio
     over time. RMS is a measure of loudness — crowd roars, commentator
     excitement, and boundaries all create energy spikes.
   → Returns a list of {timestamp_sec, energy, excitement_score} dicts.

Why audio spikes?
   The crowd and commentator voices get louder during exciting moments.
   This gives us an ML-independent, physics-based signal that's very
   reliable — you don't need labelled training data at all.
"""

import gc
import os
import json
import numpy as np
import librosa          
import whisper          
import soundfile as sf  
print("Whisper version:", whisper.__version__)


#audio extraction
def extract_audio(video_path: str, output_wav: str) -> str:
    """
    Extract the audio track from a video file and save it as a WAV.

    We use FFmpeg (via os.system) for this — it's the most reliable
    cross-platform way to do it without loading the whole video into memory.

    Returns the path to the WAV file.
    """
    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

    cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f'-vn -acodec pcm_s16le -ar 16000 -ac 1 '   # mono 16kHz — Whisper likes this
        f'"{output_wav}" -loglevel error'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"FFmpeg failed to extract audio from {video_path}")

    print(f"[audio_utils] Audio extracted → {output_wav}")
    return output_wav


#transcribing audio
def transcribe_audio(wav_path: str, model_size: str = "base") -> list[dict]:
    """
    Transcribe speech in a WAV file using OpenAI Whisper.

    model_size options:
        "tiny"    → fastest, least accurate (~32MB)
        "base"    → good balance for commentary (~74MB)   ← DEFAULT
        "small"   → better, slower (~244MB)
        "large-v3"→ best, very slow (~1.5GB)

    Returns a list of segment dicts:
        [
          {"start": 0.0, "end": 3.5, "text": "That's a SIX! Massive hit!"},
          {"start": 3.5, "end": 7.0, "text": "The fielder ran hard..."},
          ...
        ]
    """
    print(f"[audio_utils] Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    print(f"[audio_utils] Transcribing {wav_path} ...")
    result = model.transcribe(wav_path, verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end":   round(seg["end"],   2),
            "text":  seg["text"].strip(),
        })

    print(f"[audio_utils] Transcription done — {len(segments)} segments")
    del model  # Free up VRAM
    gc.collect()     # This forces Python to instantly empty the RAM
    return segments


#audio energy
def detect_audio_spikes(
    wav_path:         str,
    window_sec:       float = 1.0,
    top_n_percentile: float = 85.0,
) -> list[dict]:
    """
    Analyse the audio waveform to find moments of high energy (excitement).

    How it works:
    ─────────────
    1. Load the audio as a 1D array of amplitude values.
    2. Split it into windows of `window_sec` seconds each.
    3. Compute the RMS (Root Mean Square) energy for each window.
        RMS is essentially the "loudness" of that window.
    4. Normalise the RMS values to a 0-1 range → this is excitement_score.
    5. Flag windows above the `top_n_percentile` as "spikes".

    Parameters
    ----------
    wav_path         : path to the WAV file
    window_sec       : length of each analysis window in seconds (default 1s)
    top_n_percentile : windows in the top X% of energy are flagged as spikes

    Returns
    -------
    List of dicts for ALL windows (not just spikes):
        [
          {
            "timestamp_sec":  12.0,
            "energy":         0.043,        # raw RMS value
            "excitement_score": 0.87,       # normalised 0-1
            "is_spike":       True,
          },
          ...
        ]
    """
    print(f"[audio_utils] Analysing audio energy in {wav_path} ...")

    y, sr = librosa.load(wav_path, sr=None, mono=True)

    hop_length    = int(window_sec * sr)
    frame_length  = hop_length * 2         # overlap for smoother results

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )

    rms_min, rms_max = rms.min(), rms.max()
    if rms_max - rms_min < 1e-8:
        # Prevent divide-by-zero if audio is nearly silent
        norm_rms = np.zeros_like(rms)
    else:
        norm_rms = (rms - rms_min) / (rms_max - rms_min)

    threshold = np.percentile(norm_rms, top_n_percentile)

    results = []
    for i, (t, raw_e, norm_e) in enumerate(zip(times, rms, norm_rms)):
        results.append({
            "timestamp_sec":    round(float(t),      2),
            "energy":           round(float(raw_e),  5),
            "excitement_score": round(float(norm_e), 4),
            "is_spike":         bool(norm_e >= threshold),
        })

    spike_count = sum(1 for r in results if r["is_spike"])
    print(f"[audio_utils] Found {spike_count} audio spikes "
          f"(threshold={threshold:.2f}, percentile={top_n_percentile})")

    out_dir = '/content/drive/MyDrive' if os.path.exists('/content/drive/MyDrive') else 'assets'
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        audio_data = [{"time": float(t), "energy": float(e)} for t, e in zip(times, norm_rms)][::10]
        
        with open(os.path.join(out_dir, 'audio_telemetry.json'), 'w') as f:
            json.dump(audio_data, f, indent=4)
        print(f"[audio_utils] 🔊 Saved audio telemetry to {out_dir}/audio_telemetry.json")
    except Exception as e:
        print(f"[audio_utils] ⚠️ Could not save telemetry: {e}")

    return results


def get_excitement_at(
    spikes:        list[dict],
    timestamp_sec: float,
    window_sec:    float = 5.0,
) -> float:
    """
    Given the output of detect_audio_spikes(), return the peak excitement
    score within a ±window_sec window around a timestamp.

    Used by the fusion layer to ask: "how excited was the crowd
    around the time of this detected event?"
    """
    relevant = [
        s["excitement_score"]
        for s in spikes
        if abs(s["timestamp_sec"] - timestamp_sec) <= window_sec
    ]
    return max(relevant) if relevant else 0.0
