
import cv2
import re
import time

from datetime import datetime
from rapidocr import RapidOCR

from src.config import (
    PREVIEW_WIDTH,
    SAVE_GPS_OVERLAY,
    GPS_DIR
)


class GPSOCR:

    def __init__(self):

        print("Initializing RapidOCR...")

        self.engine = RapidOCR()
        self.roi = None

        GPS_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # SELECT GPS OVERLAY ROI
    # ==========================================

    def select_roi(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, PREVIEW_WIDTH / width)
        preview = cv2.resize(
            frame,
            (
              int(width * scale),
              int(height * scale)
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
            raise RuntimeError("No GPS ROI selected")

        # Convert preview coordinates to original resolution
        x = round(x / scale)
        y = round(y / scale)
        w = round(w / scale)
        h = round(h / scale)

        self.roi = (x, y, w, h)

        print(f"Selected GPS ROI: {self.roi}")

        if SAVE_GPS_OVERLAY:

            overlay = frame[y:y+h, x:x+w]
            cv2.imwrite(
                str(GPS_DIR / "selected_overlay.png"),
                overlay
            )

        return self.roi

    # ==========================================
    # RUN OCR
    # ==========================================

    def run_ocr(self, image):

        start = time.perf_counter()
        result = self.engine(
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

        if texts:
            if isinstance(texts, str):
                text_parts = [texts]

            else:
                text_parts = []

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

    # ==========================================
    # PARSE OCR TEXT
    # ==========================================

    
    def parse_ocr(self, raw_text):

        data = {
            "date": None,
            "time": None,
            "speed_kmh": None,
            "latitude": None,
            "longitude": None
        }

        text = raw_text.upper().strip()

        # Correct a known OCR confusion:
        # *026-02-06 -> 2026-02-06
        text = re.sub(
            r"(?<![A-Z0-9])\*(?=026[-/]\d{2}[-/]\d{2})",
            "2",
            text
        )

        # =================================
        # DATE: Support multiple formats
        # =================================

        date_patterns = [
            (r"\b(\d{4})-(\d{2})-(\d{2})\b", "%Y-%m-%d"),
            (r"\b(\d{4})/(\d{2})/(\d{2})\b", "%Y/%m/%d"),
            (r"\b(\d{2})/(\d{2})/(\d{4})\b", "%m/%d/%Y"),
            (r"\b(\d{2})-(\d{2})-(\d{4})\b", "%d-%m-%Y")
        ]

        for pattern, date_format in date_patterns:

            match = re.search(pattern, text)

            if match:
                try:
                    parsed_date = datetime.strptime(
                        match.group(0),
                        date_format
                    )

                    data["date"] = parsed_date.strftime("%Y-%m-%d")
                    break

                except ValueError:
                    continue

        # =================================
        # TIME: HH:MM:SS
        # =================================

        time_match = re.search(
            r"(?<!\d)(\d{2}:\d{2}:\d{2})(?!\d)",
            text
        )

        if time_match:

            try:
                parsed_time = datetime.strptime(
                    time_match.group(1),
                    "%H:%M:%S"
                )

                data["time"] = parsed_time.strftime("%H:%M:%S")

            except ValueError:
                pass

        # =================================
        # SPEED: 027KM/H
        # =================================

        speed_match = re.search(
            r"(\d{1,3})\s*KM\s*/\s*H",
            text
        )

        if speed_match:

            speed = int(speed_match.group(1))

            if 0 <= speed <= 250:
                data["speed_kmh"] = speed

        # =================================
        # LATITUDE
        # =================================

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

        # =================================
        # LONGITUDE
        # =================================

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

    # ==========================================
    # PROCESS GPS ROI FROM A FRAME
    # ==========================================

    def extract(self, frame, frame_id, video_time):
        if self.roi is None:
            raise RuntimeError(
                "GPS ROI has not been selected"
            )
        x, y, w, h = self.roi

        cropped = frame[y:y+h, x:x+w]

        raw_text, confidence, ocr_time = self.run_ocr(
            cropped
        )

        parsed = self.parse_ocr(raw_text)

        gps_data = {
            "frame_id": frame_id,
            "video_time": round(video_time, 3),
            "raw_ocr": raw_text,
            **parsed,
            "confidence": round(confidence, 4),
            "ocr_time_seconds": round(ocr_time, 4)
        }

        return gps_data
