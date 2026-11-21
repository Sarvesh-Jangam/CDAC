# import cv2
# import csv

# from pathlib import Path
# from ultralytics import YOLO

# from src.config import (
#     MODEL_PATH,
#     DETECTION_DIR,
#     CONFIDENCE,
#     IMAGE_SIZE,
#     DEVICE,
#     TRACKER_CONFIG,
#     SAVE_ANNOTATED_VIDEO,
#     VIDEO_CODEC,
#     PROGRESS_INTERVAL,
#     FRAME_STRIDE
# )

# from src.video_reader import VideoReader


# CSV_COLUMNS = [
#     "frame_id",
#     "video_time",
#     "track_id",
#     "class_id",
#     "class_name",
#     "confidence",
#     "x1",
#     "y1",
#     "x2",
#     "y2"
# ]


# class VehicleDetector:

#     def __init__(self):

#         print("Loading YOLO model...")

#         self.model = YOLO(str(MODEL_PATH))

#         print("YOLO model loaded.")
#         print("Classes:", self.model.names)

#     def detect(self, frame):

#         results = self.model.track(
#             frame,
#             persist=True,
#             tracker=TRACKER_CONFIG,
#             conf=CONFIDENCE,
#             imgsz=IMAGE_SIZE,
#             device=DEVICE,
#             verbose=False
#         )

#         return results[0]

#     def process_video(self, video_path):

#         video_path = Path(video_path)

#         DETECTION_DIR.mkdir(parents=True, exist_ok=True)

#         csv_path = (
#             DETECTION_DIR /
#             f"{video_path.stem}_detections.csv"
#         )
    
#         video_output = (
#             DETECTION_DIR /
#             f"{video_path.stem}_tracked.mp4"
#         )

#         processed_frames = 0
#         detection_count = 0

#         writer = None

#         with VideoReader(video_path) as reader:

#             metadata = reader.get_metadata()

#             fps = metadata["fps"]
#             width = metadata["width"]
#             height = metadata["height"]
#             total_frames = metadata["total_frames"]

#             print("\nVideo information")
#             print("-------------------------")
#             print("FPS:", fps)
#             print(f"Resolution: {width}x{height}")
#             print("Total frames:", total_frames)

#             # Initialize annotated video writer
#             if SAVE_ANNOTATED_VIDEO:

#                 writer = cv2.VideoWriter(
#                     str(video_output),
#                     cv2.VideoWriter_fourcc(*VIDEO_CODEC),
#                     fps / FRAME_STRIDE,
#                     (width, height)
#                 )

#                 if not writer.isOpened():
#                     raise RuntimeError(
#                         "Could not initialize video writer"
#                     )

#             try:

#                 with open(
#                     csv_path,
#                     "w",
#                     newline="",
#                     encoding="utf-8"
#                 ) as file:

#                     csv_writer = csv.DictWriter(
#                         file,
#                         fieldnames=CSV_COLUMNS
#                     )

#                     csv_writer.writeheader()

#                     for frame_id, video_time, frame in reader.frames():

#                         # YOLO + ByteTrack
#                         result = self.detect(frame)

#                         processed_frames += 1

#                         # Annotate frame
#                         if writer is not None:

#                             annotated_frame = result.plot()
#                             writer.write(annotated_frame)

#                         # Extract detection information
#                         if result.boxes is not None:

#                             for box in result.boxes:

#                                 class_id = int(box.cls[0])
#                                 class_name = self.model.names[class_id]

#                                 confidence = float(box.conf[0])

#                                 x1, y1, x2, y2 = (
#                                     box.xyxy[0].tolist()
#                                 )

#                                 track_id = (
#                                     int(box.id[0])
#                                     if box.id is not None
#                                     else None
#                                 )

#                                 csv_writer.writerow({

#                                     "frame_id": frame_id,
#                                     "video_time": round(video_time, 3),

#                                     "track_id": track_id,

#                                     "class_id": class_id,
#                                     "class_name": class_name,

#                                     "confidence": round(confidence, 4),

#                                     "x1": round(x1, 2),
#                                     "y1": round(y1, 2),
#                                     "x2": round(x2, 2),
#                                     "y2": round(y2, 2)
#                                 })

#                                 detection_count += 1

#                         # Progress reporting
#                         if processed_frames % PROGRESS_INTERVAL == 0:

#                             print(
#                                 f"Processed: {processed_frames} | "
#                                 f"Original frame: {frame_id}/{total_frames} | "
#                                 f"Detections: {detection_count}"
#                             )

#             finally:

#                 if writer is not None:
#                     writer.release()

#         print("\nProcessing completed!")
#         print("-------------------------")
#         print("Frames processed:", processed_frames)
#         print("Total detections:", detection_count)
#         print("CSV output:", csv_path)

#         if SAVE_ANNOTATED_VIDEO:
#             print("Annotated video:", video_output)



import cv2
import csv

from pathlib import Path
from ultralytics import YOLO

from src.config import (
    MODEL_PATH,
    DETECTION_DIR,
    GPS_DIR,
    ANNOTATED_DIR,
    CONFIDENCE,
    IMAGE_SIZE,
    DEVICE,
    FRAME_STRIDE,
    TRACKER_CONFIG,
    TARGET_CLASSES,
    SAVE_ANNOTATED_VIDEO,
    VIDEO_CODEC,
    PROGRESS_INTERVAL
)

from src.video_reader import VideoReader
from src.gps_ocr import GPSOCR


DETECTION_COLUMNS = [
    "frame_id",
    "video_time",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
]

GPS_COLUMNS = [
    "frame_id",
    "video_time",
    "raw_ocr",
    "date",
    "time",
    "speed_kmh",
    "latitude",
    "longitude",
    "confidence",
    "ocr_time_seconds"
]


class VehicleDetector:

    def __init__(self):

        print("Loading YOLO model...")
        self.model = YOLO(str(MODEL_PATH))
        print("YOLO model loaded.")
        print("Classes:", self.model.names)

        # Initialize OCR engine once
        self.gps_ocr = GPSOCR()

    def detect(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        return results[0]

    def process_video(self, video_path):

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        # Create output directories
        for directory in (
            DETECTION_DIR,
            GPS_DIR,
            ANNOTATED_DIR
        ):
            directory.mkdir(parents=True, exist_ok=True)

        detection_csv = (
            DETECTION_DIR /
            f"{video_path.stem}_detections.csv"
        )

        gps_csv = (
            GPS_DIR /
            f"{video_path.stem}_gps.csv"
        )

        video_output = (
            ANNOTATED_DIR /
            f"{video_path.stem}_tracked.mp4"
        )

        processed_frames = 0
        detection_count = 0
        gps_count = 0

        writer = None

        with VideoReader(video_path) as reader:

            metadata = reader.get_metadata()

            fps = metadata["fps"]
            width = metadata["width"]
            height = metadata["height"]
            total_frames = metadata["total_frames"]

            print("\nVideo information")
            print("-------------------------")
            print("FPS:", fps)
            print(f"Resolution: {width}x{height}")
            print("Total frames:", total_frames)

            if SAVE_ANNOTATED_VIDEO:

                writer = cv2.VideoWriter(
                    str(video_output),
                    cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                    fps / FRAME_STRIDE,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not initialize video writer"
                    )

            try:

                with (
                    open(detection_csv, "w", newline="", encoding="utf-8") as det_file,
                    open(gps_csv, "w", newline="", encoding="utf-8") as gps_file
                ):

                    detection_writer = csv.DictWriter(
                        det_file,
                        fieldnames=DETECTION_COLUMNS
                    )

                    gps_writer = csv.DictWriter(
                        gps_file,
                        fieldnames=GPS_COLUMNS
                    )

                    detection_writer.writeheader()
                    gps_writer.writeheader()

                    for frame_id, video_time, frame in reader.frames():

                        # Select ROI once, using the first frame
                        if self.gps_ocr.roi is None:
                            self.gps_ocr.select_roi(frame)

                        # YOLO + ByteTrack
                        result = self.detect(frame)

                        processed_frames += 1

                        boxes = result.boxes

                        # Determine whether this frame contains
                        # at least one target construction vehicle
                        target_detections = []

                        if boxes is not None:

                            for box in boxes:

                                class_id = int(box.cls[0])
                                class_name = self.model.names[class_id]

                                if class_name in TARGET_CLASSES:
                                    target_detections.append(box)

                        # Run OCR only once per relevant frame
                        if target_detections:

                            try:

                                gps_data = self.gps_ocr.extract(
                                    frame,
                                    frame_id,
                                    video_time
                                )

                                gps_writer.writerow(gps_data)
                                gps_count += 1

                                print(
                                    f"[GPS] Frame {frame_id} | "
                                    f"{gps_data['latitude']}, "
                                    f"{gps_data['longitude']}"
                                )

                            except Exception as error:

                                print(
                                    f"[GPS ERROR] Frame {frame_id}: {error}"
                                )

                        # Save all model detections to detection CSV
                        if boxes is not None:

                            for box in boxes:

                                class_id = int(box.cls[0])
                                class_name = self.model.names[class_id]

                                confidence = float(box.conf[0])

                                x1, y1, x2, y2 = (
                                    box.xyxy[0].tolist()
                                )

                                track_id = (
                                    int(box.id[0])
                                    if box.id is not None
                                    else None
                                )

                                detection_writer.writerow({

                                    "frame_id": frame_id,
                                    "video_time": round(video_time, 3),

                                    "track_id": track_id,

                                    "class_id": class_id,
                                    "class_name": class_name,

                                    "confidence": round(confidence, 4),

                                    "x1": round(x1, 2),
                                    "y1": round(y1, 2),
                                    "x2": round(x2, 2),
                                    "y2": round(y2, 2)
                                })

                                detection_count += 1

                        # Save annotated frame
                        if writer is not None:

                            annotated_frame = result.plot()
                            writer.write(annotated_frame)

                        # Progress reporting
                        if processed_frames % PROGRESS_INTERVAL == 0:

                            print(
                                f"Processed: {processed_frames} | "
                                f"Original frame: {frame_id}/{total_frames} | "
                                f"Detections: {detection_count} | "
                                f"GPS OCR calls: {gps_count}"
                            )

            finally:

                if writer is not None:
                    writer.release()

        print("\nProcessing completed!")
        print("-------------------------")
        print("Frames processed:", processed_frames)
        print("Total detections:", detection_count)
        print("GPS OCR calls:", gps_count)
        print("Detection CSV:", detection_csv)
        print("GPS CSV:", gps_csv)

        if SAVE_ANNOTATED_VIDEO:
            print("Annotated video:", video_output)
