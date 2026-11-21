
import cv2
import re
import csv
import time
from datetime import datetime
from rapidocr import RapidOCR

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = (
    "/home/bruce-wayne/Downloads/TAR_Vids_dashcam/"
    "no_audio_20250420190907_002946F.MP4"
)

OUTPUT_CSV = "ocr/gps_extraction_optimized.csv"
OUTPUT_OVERLAY = "ocr/selected_overlay.png"

FRAME_INTERVAL_SECONDS = 1
PREVIEW_WIDTH = 1200

# ============================================================
# INITIALIZE
# ============================================================

print("Initializing RapidOCR...")

engine = RapidOCR()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = total_frames / fps

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Resolution: {original_width} x {original_height}")
print(f"Duration: {duration:.2f} seconds")

# ============================================================
# SELECT GPS OVERLAY REGION
# ============================================================

success, first_frame = cap.read()

if not success:
    raise RuntimeError("Could not read first frame")

scale = min(1.0, PREVIEW_WIDTH / original_width)

preview = cv2.resize(
    first_frame,
    (
        int(original_width * scale),
        int(original_height * scale)
    )
)

print("\nSelect the complete GPS overlay.")
print("Press ENTER or SPACE to confirm.")

roi = cv2.selectROI(
    "Select GPS Overlay",
    preview,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x, y, w, h = roi

if w == 0 or h == 0:
    cap.release()
    raise RuntimeError("No region selected")

# Convert coordinates to original resolution
x = round(x / scale)
y = round(y / scale)
w = round(w / scale)
h = round(h / scale)

print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")

# Save overlay for reference
overlay = first_frame[y:y+h, x:x+w]
cv2.imwrite(OUTPUT_OVERLAY, overlay)

# ============================================================
# RECOGNITION-ONLY OCR
# ============================================================

def run_ocr(image):

    start = time.perf_counter()

    result = engine(
        image,
        use_det=False,
        use_cls=False,
        use_rec=True
    )

    elapsed = time.perf_counter() - start

    if result is None:
        return "", 0.0, elapsed

    raw_text = ""
    confidence = 0.0

    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)

    # Handle different RapidOCR result representations
    if texts:
        text_parts = []

        if isinstance(texts, str):
            text_parts = [texts]

        else:
            for item in texts:
                if isinstance(item, (tuple, list)):
                    if item:
                        text_parts.append(str(item[0]))
                        if len(item) > 1:
                            try:
                                confidence = max(
                                    confidence,
                                    float(item[1])
                                )
                            except (ValueError, TypeError):
                                pass
                else:
                    text_parts.append(str(item))

        raw_text = " ".join(text_parts).strip()

    # Read confidence if supplied separately
    if scores is not None:
        if isinstance(scores, (int, float)):
            confidence = max(confidence, float(scores))

        else:
            try:
                confidence = max(
                    confidence,
                    max(float(s) for s in scores)
                )
            except (ValueError, TypeError):
                pass

    return raw_text, confidence, elapsed


# ============================================================
# PARSE OCR TEXT
# ============================================================

def parse_ocr(raw_text):

    data = {
        "date": None,
        "time": None,
        "speed_kmh": None,
        "latitude": None,
        "longitude": None
    }

    text = raw_text.upper()

    # DATE: MM/DD/YYYY
    date_match = re.search(
        r"\b(\d{2}/\d{2}/\d{4})\b",
        text
    )

    if date_match:
        try:
            datetime.strptime(
                date_match.group(1),
                "%m/%d/%Y"
            )
            data["date"] = date_match.group(1)
        except ValueError:
            pass

    # TIME: HH:MM:SS
    time_match = re.search(
        r"(?<!\d)(\d{2}:\d{2}:\d{2})(?!\d)",
        text
    )

    if time_match:
        try:
            datetime.strptime(
                time_match.group(1),
                "%H:%M:%S"
            )
            data["time"] = time_match.group(1)
        except ValueError:
            pass

    # SPEED: 027KM/H
    speed_match = re.search(
        r"(\d{1,3})\s*KM\s*/\s*H",
        text
    )

    if speed_match:
        speed = int(speed_match.group(1))

        if 0 <= speed <= 250:
            data["speed_kmh"] = speed

    # LATITUDE: N:17.5219
    # OCR can confuse decimal point with -, / or ,
    lat_match = re.search(
        r"N\s*[:=]?\s*(\d{1,2})[.,/\-](\d{3,7})",
        text
    )

    if lat_match:
        latitude = float(
            lat_match.group(1) + "." + lat_match.group(2)
        )

        if 0 <= latitude <= 90:
            data["latitude"] = latitude

    # LONGITUDE: E:73.5392
    lon_match = re.search(
        r"E\s*[:=]?\s*(\d{1,3})[.,/\-](\d{3,7})",
        text
    )

    if lon_match:
        longitude = float(
            lon_match.group(1) + "." + lon_match.group(2)
        )

        if 0 <= longitude <= 180:
            data["longitude"] = longitude

    return data


# ============================================================
# PROCESS VIDEO SEQUENTIALLY
# ============================================================

print("\nStarting optimized OCR extraction...\n")

results = []

frame_number = 0
next_ocr_frame = 0

total_ocr_time = 0.0
successful_frames = 0

video_start = time.perf_counter()

# Process the already extracted first frame as frame 0,
# then continue sequentially from frame 1.
while True:

    if frame_number == 0:
        frame = first_frame
        success = True
    else:
        success, frame = cap.read()

    if not success:
        break

    if frame_number >= next_ocr_frame:

        video_second = frame_number / fps

        cropped = frame[y:y+h, x:x+w]

        try:
            raw_text, confidence, ocr_time = run_ocr(cropped)

            total_ocr_time += ocr_time

            parsed = parse_ocr(raw_text)

            row = {
                "video_time_seconds": round(video_second, 3),
                "frame_number": frame_number,
                "raw_ocr": raw_text,
                **parsed,
                "confidence": round(confidence, 4),
                "ocr_time_seconds": round(ocr_time, 4)
            }

            results.append(row)

            if (
                parsed["latitude"] is not None
                and parsed["longitude"] is not None
            ):
                successful_frames += 1

            print(
                f"[{video_second:05.1f}s] "
                f"GPS: {parsed['latitude']}, "
                f"{parsed['longitude']} | "
                f"OCR: {ocr_time:.3f}s | "
                f"Confidence: {confidence:.3f}"
            )

        except Exception as error:

            print(f"[{video_second:.1f}s] OCR error: {error}")

            results.append({
                "video_time_seconds": round(video_second, 3),
                "frame_number": frame_number,
                "raw_ocr": "",
                "date": None,
                "time": None,
                "speed_kmh": None,
                "latitude": None,
                "longitude": None,
                "confidence": 0.0,
                "ocr_time_seconds": 0.0
            })

        next_ocr_frame += round(fps * FRAME_INTERVAL_SECONDS)

    frame_number += 1

cap.release()

total_processing_time = time.perf_counter() - video_start

# ============================================================
# EXPORT CSV
# ============================================================

fieldnames = [
    "video_time_seconds",
    "frame_number",
    "raw_ocr",
    "date",
    "time",
    "speed_kmh",
    "latitude",
    "longitude",
    "confidence",
    "ocr_time_seconds"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

# ============================================================
# SUMMARY
# ============================================================

print("\n========== EXTRACTION SUMMARY ==========")

print(f"Frames processed: {len(results)}")
print(f"Complete GPS readings: {successful_frames}")
print(f"Incomplete GPS readings: {len(results) - successful_frames}")

print(f"\nTotal OCR inference time: {total_ocr_time:.2f} seconds")
print(f"Total processing time: {total_processing_time:.2f} seconds")

if results:
    print(
        f"Average OCR time per frame: "
        f"{total_ocr_time / len(results):.3f} seconds"
    )

print(f"\nCSV saved: {OUTPUT_CSV}")
print(f"Overlay saved: {OUTPUT_OVERLAY}")
