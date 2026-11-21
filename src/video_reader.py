
import cv2

from src.config import FRAME_STRIDE


class VideoReader:

    def __init__(self, video_path):

        self.video_path = str(video_path)
        
        if FRAME_STRIDE < 1:
            raise ValueError("FRAME_STRIDE must be at least 1")
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if (
            self.fps <= 0
            or self.width <= 0
            or self.height <= 0
        ):
            self.cap.release()
            raise RuntimeError("Invalid video properties")

    def get_metadata(self):

        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames
        }

    def frames(self):

        frame_id = 0

        # Read first frame: frame 0
        success, frame = self.cap.read()

        while success:

            current_frame = frame_id
            video_time = current_frame / self.fps

            yield current_frame, video_time, frame

            # Skip frames and retrieve the next selected frame
            for _ in range(FRAME_STRIDE):

                if not self.cap.grab():
                    success = False
                    break

                frame_id += 1

            if success:
                success, frame = self.cap.retrieve()

    def release(self):

        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
