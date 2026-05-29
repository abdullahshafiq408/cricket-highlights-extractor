# cricket-highlights-extractor
# Multimodal Cricket Highlight Engine

An autonomous Machine Learning pipeline designed to extract high-impact cricket highlights from raw broadcast video. 

By synthesizing multimodal signals—including Cricsheet JSON metadata, Scoreboard OCR, acoustic physics, semantic NLP classification, and spatial object detection—the system dynamically calculates trailing-edge synchronization to trim and export highlight clips without human intervention.

## Architectural Layers

1. **Layer 1: Ground Truth Metadata:** Parsed Cricsheet JSON for match state alignment.
2. **Layer 2: Visual Telemetry:** EasyOCR deployed to track broadcast scoreboards and detect timeline anchors (e.g., overs/balls).
3. **Layer 3: Acoustic Signal Processing:** Librosa RMS energy thresholding to detect crowd roar and impact anomalies.
4. **Layer 4: Semantic Classification:** OpenAI Whisper speech-to-text paired with a Logistic Regression classifier to evaluate commentator excitement (`P(y) ≥ 0.85`).
5. **Layer 5: Multimodal Synchronization:** Trailing-edge fusion engine that aligns lagging visual graphics with audio/NLP nodes to execute dynamic FFmpeg video trimming.
6. **Layer 6: Spatial Object Detection (Experimental):** YOLOv8 bounding-box heuristics to calculate visual action scores based on player density, ball detection, and spatial clustering.

## Tech Stack
* **Audio/NLP:** `librosa`, `whisper`
* **Computer Vision:** `opencv-python`, `easyocr`, `ultralytics` (YOLOv8)
* **Data/Math:** `numpy`, `pandas`, `scikit-learn`
* **Video Processing:** `ffmpeg`

## System Limitations
* **Hardware Acceleration:** While fully functional on standard CPUs, end-to-end processing is optimized for GPU environments, which yield an approximate 10x performance increase.
* **Format Compatibility:** The current parsing logic is calibrated strictly for limited-overs formats (T20/ODI). Multi-day Test match integration is planned for future architecture updates.
* **Broadcast Variations:** The OCR region-of-interest targets modern, bottom-aligned graphics. Legacy layouts (e.g., top-aligned Australian scoreboards) or significantly degraded, low-resolution video feeds may reduce extraction accuracy.
