import cv2
import numpy as np
import time

from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

class VisualProcessor:
    def __init__(self, config):
        # 前端切换关键帧参数
        self.config = config
        self.max_features_to_detect = self.config.get('max_features_to_detect', 1000) # 最大特征点数
        self.min_parallax = self.config.get('min_parallax', 10) # 最小视差
        self.min_track_ratio = self.config.get('min_track_ratio', 0.8) # 最小跟踪比例
        self.visualize_flag = self.config.get('visualize', True) # 是否可视化追踪结果

        # 读取相机内参
        cam_matrix_raw = self.config.get('cam_intrinsics', np.eye(3).flatten().tolist())
        self.cam_matrix = np.asarray(cam_matrix_raw).reshape(3, 3)

        # 读取相机畸变参数
        dist_coeffs_raw = self.config.get('distortion_coefficients', np.zeros(4).tolist())
        self.dist_coeffs = np.asarray(dist_coeffs_raw).reshape(4, 1)

        self.min_dist = self.config.get('min_dist', 15) # 特征点间的最小像素距离

        # 特征点id
        self.next_feature_id = 0
        self.prev_pt_ids = None

        self.prev_pts = None
        self.prev_gray = None

    # 提取特征点
    def detect_features(self, gray_image, max_corners, mask=None):
        return cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=self.min_dist,
            blockSize=7,
            mask=mask
        )

    def undistort_points(self, points):
        if self.cam_matrix is None or self.dist_coeffs is None or len(points) == 0:
            return points # 如果没有内参，则返回原始点
        
        # cv2.undistortPoints 需要 (N, 1, 2) 的形状
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        
        # P设置为K，可以直接输出去畸变后的像素坐标
        undistorted_points = cv2.undistortPoints(points_reshaped, self.cam_matrix, self.dist_coeffs, P=self.cam_matrix)
        
        return undistorted_points.reshape(-1, 2) # 返回 (N, 2) 形状

    # 光流追踪特征点
    def track_features(self, image):
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 一些局部变量的初始化
        is_kf = False
        new_pts = None
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_gray = curr_gray
            self.prev_pts = self.detect_features(self.prev_gray, self.max_features_to_detect)
            if self.prev_pts is None:
                return None, None, False

            # 分配初始ID
            num_new_features = len(self.prev_pts)
            self.prev_pt_ids = np.arange(self.next_feature_id, self.next_feature_id + num_new_features)
            self.next_feature_id += num_new_features

            undistorted_pts = self.undistort_points(self.prev_pts)

            # --- 可视化第一帧的ID ---
            if self.visualize_flag:
                vis_img = image.copy()
                for pt, feature_id in zip(self.prev_pts, self.prev_pt_ids):
                    pos = tuple(pt.ravel().astype(int))
                    cv2.circle(vis_img, pos, 3, (0, 255, 0), -1)
                    cv2.putText(vis_img, str(feature_id), (pos[0]+5, pos[1]-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.imshow("Optical Flow", vis_img)
                cv2.waitKey(1)

            return undistorted_pts, self.prev_pt_ids, True
                
        # 正向光流
        curr_pts, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # 反向光流
        prev_pts_backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray, curr_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        forward_status = forward_status.flatten()

        # 反向光流误差
        # 将数组从 (N, 1, 2) reshape 为 (N, 2) 以便正确计算距离
        prev_pts_reshaped = self.prev_pts.reshape(-1, 2)
        prev_pts_backward_reshaped = prev_pts_backward.reshape(-1, 2)
        fb_error = np.linalg.norm(prev_pts_reshaped - prev_pts_backward_reshaped, axis=1)

        # 最终掩码
        final_mask = (forward_status == 1) & (fb_error < 1.0)

        # 筛选内点
        good_prev = self.prev_pts[final_mask]
        good_curr = curr_pts[final_mask]
        good_ids = self.prev_pt_ids[final_mask]

        # Measure displacement to mean parallax
        displacement = np.linalg.norm(good_curr.reshape(-1, 2) - good_prev.reshape(-1, 2), axis=1)
        mean_parallax = np.mean(displacement) if len(displacement) > 0 else 0
        if mean_parallax > self.min_parallax:
            is_kf = True

        num_current_features = len(good_curr)

        final_pts = good_curr
        final_ids = good_ids
        
        # 可视化，保持数组长度一致
        if self.visualize_flag:
            vis_img = image.copy()
            # 同时遍历旧位置、新位置和ID
            for p1, p2, feature_id in zip(good_prev, good_curr, good_ids):
                p1_t = tuple(p1.ravel().astype(int))
                p2_t = tuple(p2.ravel().astype(int))
                
                # 画出光流轨迹
                cv2.arrowedLine(vis_img, p1_t, p2_t, color=(0, 255, 0), thickness=1, tipLength=0.3)
                
                # 画出特征点的ID
                # 参数: 图像, 文本, 左下角坐标, 字体, 大小, 颜色, 线宽
                cv2.putText(vis_img, str(feature_id), (p2_t[0]+5, p2_t[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.putText(vis_img, f"Tracked: {len(good_ids)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Optical Flow", vis_img)
            cv2.waitKey(1)

        # 如果跟踪到的特征点太少，则补充特征点
        if len(good_curr) < (self.max_features_to_detect * self.min_track_ratio):
            is_kf = True
            mask = np.ones_like(curr_gray)
            for pt in good_curr.reshape(-1, 2):
                cv2.circle(mask, tuple(pt.astype(int)), self.min_dist, 0, -1)

            num_new_features_needed = self.max_features_to_detect - num_current_features
            
            new_pts = self.detect_features(curr_gray, num_new_features_needed, mask=mask)

            if new_pts is not None:
                # 分配新的特征点id
                num_new = len(new_pts)
                new_ids = np.arange(self.next_feature_id, self.next_feature_id + num_new)
                self.next_feature_id += num_new

                final_pts = np.vstack([good_curr, new_pts])
                final_ids = np.hstack([good_ids, new_ids])
        
        self.prev_gray = curr_gray
        self.prev_pts = final_pts
        self.prev_pt_ids = final_ids

        # 特征点去畸变
        undistorted_final_pts = self.undistort_points(final_pts)

        return undistorted_final_pts, final_ids, is_kf