from src.detector import VehicleDetector
from src.config import VIDEO_DIR


def main():

    # video_path = VIDEO_DIR / "Road construction __ Dashcam footage 48 #travel_480p.mp4"
    video_path = "/home/bruce-wayne/Downloads/TAR_Vids_dashcam/no_audio_20250415161214_002499F.MP4"

    detector = VehicleDetector()

    detector.process_video(video_path)


if __name__ == "__main__":
    main()
