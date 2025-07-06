import cv2
import numpy as np
from collections import deque
import time
import logging
import argparse
import os

class WagonDetectorConfig:
    def __init__(self):
        self.resize_width = 1920
        self.resize_height = 1080
        self.blur_kernel_size = 21
        self.history = 500
        self.var_threshold = 16
        self.detect_shadows = False
        self.min_area = 3000
        self.line_position = 0.4
        self.line_threshold = 10
        self.min_frames_between_counts = 15
        self.tracker_length = 50

class WagonDetector:
    def __init__(self, config):
        self.config = config
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.detect_shadows
        )
        self.wagon_count = 0
        self.crossed_objects = set()
        self.object_tracker = deque(maxlen=config.tracker_length)
        self.last_count_frame = 0
        self.roi_points = None
        self.roi_mask = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('WagonDetector')

    def setup_roi(self, frame):
        height, width = frame.shape[:2]
        x_offset = 400
        y_offset = 400

        self.roi_points = np.array([
            [int(width * 0.1) + x_offset, int(height * 0.4) + y_offset],
            [int(width * 0.9) + x_offset, int(height * 0.4) + y_offset],
            [int(width * 0.95) + x_offset, int(height * 0.8) + y_offset],
            [int(width * 0.05) + x_offset, int(height * 0.8) + y_offset]
        ], np.int32)

        self.roi_points = np.clip(self.roi_points, [0, 0], [width-1, height-1])
        self.roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_points], 255)

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (self.config.resize_width, self.config.resize_height))

        if self.roi_mask is None:
            self.setup_roi(frame)

        roi_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        blurred = cv2.GaussianBlur(roi_frame, 
                                   (self.config.blur_kernel_size, self.config.blur_kernel_size), 
                                   0)
        return blurred

    def detect_motion(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        return fg_mask

    def process_frame(self, frame, frame_count):
        processed = self.preprocess_frame(frame)
        height, width = processed.shape[:2]
        motion_mask = self.detect_motion(processed)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.polylines(frame, [self.roi_points], True, (0, 255, 255), 2)

        x_offset = 400
        y_offset = 400
        line_y = int(height * self.config.line_position) + y_offset
        cv2.line(frame, 
                 (int(width * 0.1) + x_offset, line_y), 
                 (int(width * 0.9) + x_offset, line_y), 
                 (0, 255, 0), 2)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.min_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if cv2.pointPolygonTest(self.roi_points, (cx, cy), False) >= 0:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        if (abs(cy - line_y) < self.config.line_threshold and
                            frame_count - self.last_count_frame > self.config.min_frames_between_counts):
                            object_id = f"{cx}_{cy}"
                            if object_id not in self.crossed_objects:
                                self.wagon_count += 1
                                self.crossed_objects.add(object_id)
                                self.last_count_frame = frame_count
                                self.logger.info(f"Wagon detected! Count: {self.wagon_count}")

                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Wagon Count: {self.wagon_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame

def process_video(video_path, output_path=None, display=False):
    config = WagonDetectorConfig()
    detector = WagonDetector(config)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    processing_times = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            processed_frame = detector.process_frame(frame, frame_count)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            if output_path:
                out.write(processed_frame)

            if display:
                cv2.imshow('Wagon Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")

    finally:
        if processing_times:
            avg_processing_time = np.mean(processing_times)
            avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            print(f"\nPerformance Metrics:")
            print(f"Average processing time per frame: {avg_processing_time:.3f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

    return detector.wagon_count