from pathlib import Path

# =========================
# Configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
VIDEO_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DETECTION_DIR = OUTPUT_DIR / "detections"
GPS_DIR = OUTPUT_DIR / "gps"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"

# YOLO configuration
CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEVICE = "cpu"

# Target construction vehicle classes
TARGET_CLASSES = {
    "Excavator",
    "Dump Truck",
    "Mixer Truck",
    "Roller"
}

# Annotated video
SAVE_ANNOTATED_VIDEO = True
VIDEO_CODEC = "mp4v"

# Video configuration
FRAME_STRIDE = 5
TRACKER_CONFIG = "bytetrack.yaml"

# Progress reporting
PROGRESS_INTERVAL = 20

# GPS OCR configuration
PREVIEW_WIDTH = 1200
SAVE_GPS_OVERLAY = True