# import time
# import cv2
# import os
# # print(os.getcwd())
# def benchmark(video_path, use_grab, stride=5):

#     cap = cv2.VideoCapture(video_path)
#     print("Opened:", cap.isOpened())
#     start = time.perf_counter()
#     retrieved = 0
#     frame_id = 0

#     while True:

#         if use_grab:
#             for _ in range(stride - 1):
#                 if not cap.grab():
#                     break
#                 frame_id += 1
#             else:
#                 success, frame = cap.read()
#                 if not success:
#                     break
#                 retrieved += 1
#                 frame_id += 1
#                 continue

#             break

#         else:
#             success, frame = cap.read()

#             if not success:
#                 break

#             if frame_id % stride == 0:
#                 retrieved += 1

#             frame_id += 1

#     elapsed = time.perf_counter() - start
#     cap.release()

#     return elapsed, retrieved


# path = "data/Road Construction __ Dashcam Footage 38 #travel_720p.mp4"

# print("Read + skip:", benchmark(path, False))
# print("Grab + retrieve:", benchmark(path, True))