import cv2
import numpy as np
import time
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

class VisualProcessor:
    def __init__(self, config):
        self.config = config

        self.prev_kf_pts = None
        self.prev_gray = None
        self.keyframe_buffer_size = 3
        self.keyframe_buffer = []


    # 提取特征点
    def detect_features(self, gray_image, max_corners=1000, mask=None):
        return cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
            mask=mask
        )

    # 光流追踪特征点
    def track_features(self, image, visualize=False):
        if self.prev_gray is None or self.prev_kf_pts is None:
            self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.prev_kf_pts = self.detect_features(self.prev_gray)
            return self.prev_kf_pts, 0
        
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_kf_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        status = status.flatten()
        good_prev = self.prev_kf_pts[status == 1]
        good_curr = curr_pts[status == 1]

        # Measure displacement to mean parallax
        displacement = np.linalg.norm(good_curr - good_prev, axis=1)
        mean_parallax = np.mean(displacement) if len(displacement) > 0 else 0

        if visualize:
            vis = image.copy()
            for p1, p2 in zip(good_prev, good_curr):
                p1 = tuple(p1.ravel().astype(int))
                p2 = tuple(p2.ravel().astype(int))
                cv2.arrowedLine(vis, p1, p2, color=(0, 255, 0), thickness=1, tipLength=0.3)
            cv2.imshow("Optical Flow", vis)
            cv2.waitKey(1)
        
        self.prev_gray = curr_gray
        self.prev_kf_pts = good_curr

        return good_curr, mean_parallax


    def add_new_keyframe(self, image_path, timestamp):
        print(f"New keyframe. Adding '{image_path}' to buffer.")
        self.keyframe_buffer.append((image_path, timestamp))

        predictions = None
        if len(self.keyframe_buffer) == self.keyframe_buffer_size:
            print("Buffer full. Running model prediction.")
            # --- 滑动窗口 ---
            self.keyframe_buffer = self.keyframe_buffer[-1:]
            print("--- Keyframe buffer slided. ---")

        return predictions