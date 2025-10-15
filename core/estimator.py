from collections import deque
from ctypes import alignment
from re import S
from tkinter import N
import gtsam
import cv2
import numpy as np
import threading
import queue
from collections import deque

from .backend import Backend
from datatype.keyframe import KeyFrame
from datatype.global_map import GlobalMap
from .imu_process import IMUProcessor
from .sfm_processor import SfMProcessor
from .viewer import Viewer3D
from .vio_initializer import VIOInitializer


class Estimator(threading.Thread):
    """
    The central coordinator for the SLAM system. Runs as a consumer thread.
    """
    def __init__(self, config, input_queue, viewer_queue, global_central_map):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = input_queue
        self.global_map = global_central_map

        self.imu_processor = IMUProcessor(config)
        self.imu_buffer = []

        self.backend = Backend(global_central_map, config, self.imu_processor)

        # 读取相机内参
        cam_intrinsics_raw = self.config.get('cam_intrinsics', np.eye(3).flatten().tolist())
        self.cam_intrinsics = np.asarray(cam_intrinsics_raw).reshape(3, 3)

        self.sfm_processor = SfMProcessor(self.cam_intrinsics)

        self.next_kf_id = 0
        self.keyframe_window = deque(maxlen=10)
        self.landmarks = {}

        # 初始化相关设置
        self.is_initialized = False

        self.init_window_size = self.config.get('init_window_size', 10)

        self.gravity_magnitude = self.config.get('gravity', 9.81)
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)

        # 可视化test
        self.viewer_queue = viewer_queue

        # Threading control
        self.is_running = False

    def start(self):
        self.is_running = True
        super().start()

    def shutdown(self):
        self.is_running = False
        print("【Estimator】shut down.")

    def run(self):
        print("【Estimator】thread started.")
        while self.is_running:
            try:
                package = self.input_queue.get(timeout=1.0)

                if package is None:
                    print("【Estimator】received shutdown signal from frontend.")
                    break 

                timestamp = package['timestamp']

                # 接收IMU数据
                if 'imu_measurements' in package:
                    self.imu_buffer.append(package)

                # 接收视觉特征点数据
                elif 'visual_features' in package:
                    timestamp = package['timestamp']
                    visual_features = package['visual_features']
                    feature_ids = package['feature_ids']
                    image = package['image']

                    new_id = self.next_kf_id
                    new_kf = KeyFrame(new_id, timestamp)
                    new_kf.add_visual_features(visual_features, feature_ids)
                    new_kf.set_image(image)

                    self.next_kf_id += 1
                    self.keyframe_window.append(new_kf)

                    if not self.is_initialized:
                        if len(self.keyframe_window) == self.init_window_size:
                            self.visual_inertial_initialization()
                        
                        else:
                            print(f"【Init】: Collecting frames... {len(self.keyframe_window)}/{self.init_window_size}")
                    else:
                        pass
                        # self.process_package_data(new_kf)

            except queue.Empty:
                continue
        
        print("【Estimator】thread has finished.")

    def create_imu_factors(self, kf_start, kf_end):
        start_ts = kf_start.get_timestamp()
        end_ts = kf_end.get_timestamp()

        # 获取IMU量测数据
        measurements_with_ts = [
            (pkg['timestamp'], pkg['imu_measurements']) for pkg in self.imu_buffer
            if start_ts < pkg['timestamp'] <= end_ts
        ]

        if not measurements_with_ts:
            print(f"【Estimator】: No IMU measurements between KF {kf_start.get_id()} and KF {kf_end.get_id()}.")
            return None

        imu_preintegration = self.imu_processor.pre_integration(measurements_with_ts, start_ts, end_ts)

        if imu_preintegration:
            return {
                'start_kf_timestamp': start_ts,
                'end_kf_timestamp': end_ts,
                'imu_measurements': measurements_with_ts,
                'imu_preintegration': imu_preintegration
            }

        return None
        
    # TODO:def check_motion_excitement(self):


    def visual_inertial_initialization(self):
        print("【Init】: Buffer is full. Starting initialization process.")

        initial_keyframes = list(self.keyframe_window)
        sfm_success, ref_kf, curr_kf, ids_best, p1_best, p2_best = \
            self.visual_initialization(initial_keyframes)

        # 视觉初始化失败，滑动窗口继续初始化
        if not sfm_success:
            print("【Init】: Visual initialization failed. Sliding window.")
            return

        # 创建初始化IMU因子
        initial_imu_factors = []
        for i in range(len(initial_keyframes) - 1):
            kf_start = initial_keyframes[i]
            kf_end = initial_keyframes[i + 1]
            imu_factors = self.create_imu_factors(kf_start, kf_end)
            if imu_factors:
                initial_imu_factors.append(imu_factors)

        # 视觉惯性初始化
        alignment_success, scale, gyro_bias, velocities, gravity_w = VIOInitializer.initialize(
            initial_keyframes, 
            initial_imu_factors, 
            self.imu_processor, 
            self.gravity_magnitude, 
            self.T_bc
        )

        if alignment_success:
            # 重三角化地图点
            # 获取最终的位姿T_wc
            final_pose_ref = ref_kf.get_global_pose()
            final_pose_curr = curr_kf.get_global_pose()

            # 获取具有尺度的T_curr_ref
            final_T_curr_ref = np.linalg.inv(final_pose_curr) @ final_pose_ref
            final_R, final_t = final_T_curr_ref[:3, :3], final_T_curr_ref[:3, 3].reshape(3, 1)

            # 恢复具有尺度的3d landmark，相对于ref_kf的坐标
            final_points_3d_in_ref_frame, final_mask = self.sfm_processor.triangulate_points(p1_best, p2_best, final_R, final_t)

            # 转换到世界坐标系
            points_3d_world = (final_pose_ref[:3, :3] @ final_points_3d_in_ref_frame.T + final_pose_ref[:3, 3].reshape(3, 1)).T

            # 加入地图
            valid_ids = np.array(ids_best)[final_mask]
            self.landmarks.clear()
            for landmark_id, landmark_pt in zip(valid_ids, points_3d_world):
                self.landmarks[landmark_id] = landmark_pt

            print(f"【Init】: Re-triangulation complete. Final map has {len(self.landmarks)} landmarks.")
            print("【Init】: Alignment successful. Calling backend to build initial graph...")

            # 更新IMU偏置
            initial_bias_obj = gtsam.imuBias.ConstantBias(np.zeros(3), gyro_bias)
            self.imu_processor.update_bias(initial_bias_obj)
            
            # 进行初始优化
            self.backend.optimize(initial_keyframes, initial_imu_factors, self.landmarks, velocities, initial_bias_obj)

            self.is_initialized = True

            if self.viewer_queue:
                print("【Init】: Sending initialization result to viewer queue...")

                # VIOInitializer已经更新了所有KF的位姿
                poses = {kf.get_id(): kf.get_global_pose() for kf in initial_keyframes if kf.get_global_pose() is not None}

                vis_data = {
                    'landmarks': self.landmarks.copy(), # self.landmarks 已经是重三角化后的精确结果
                    'poses': poses
                }

                try:
                    self.viewer_queue.put_nowait(vis_data)
                except queue.Full:
                    print("【Estimator】: Viewer queue is full, skipping this frame.")
        else:
            print("【Init】: V-I Alignment failed.")

        # viewer可视化
        # if sfm_success:
        #     self._verify_landmarks_on_images(self.init_kf_buffer, self.landmarks)
        #     if self.viewer_queue:
        #         print("【Init】: Sending initialization result to viewer queue...")
        #         poses = {kf.get_id(): kf.get_global_pose() for kf in self.keyframes.values() if kf.get_global_pose() is not None}
                
        #         vis_data = {
        #             'landmarks': self.landmarks.copy(),
        #             'poses': poses
        #         }
                
        #         # 使用 try-except 避免队列满时阻塞
        #         try:
        #             self.viewer_queue.put_nowait(vis_data)
        #         except queue.Full:
        #             print("【Estimator】: Viewer queue is full, skipping this frame.")

        # if sfm_success:
        #     self.init_kf_buffer.clear() # 初始化成功后清空缓冲区
        # else:
        #     self.init_kf_buffer.pop(0) # 失败则滑动窗口
        # viewer可视化

        return alignment_success
    
    def visual_initialization(self, initial_keyframes):
        print("【Visual Init】: Searching for the best keyframe pair...")
        ref_kf = initial_keyframes[0]
        ref_kf.set_global_pose(np.eye(4))

        curr_kf = None

        R, t, inlier_ids, pts1_inliers, pts2_inliers = [None] * 5

        # 从最新的KF开始，向前找到最优的KF对
        for i in range(1, len(initial_keyframes)):
            potential_curr_kf = initial_keyframes[i]

            success, ids_cand, p1_cand, p2_cand, R_cand, t_cand = \
            self.sfm_processor.epipolar_compute(ref_kf, potential_curr_kf)

            if not success:
                continue

            parallax = np.median(np.linalg.norm(p1_cand - p2_cand, axis=1))

            # 保证一定的视差，这里是一个非常敏感的参数
            if parallax > 50:
                print(f"【Visual Init】: Found a good pair! (KF {ref_kf.get_id()}, KF {potential_curr_kf.get_id()}) "
                      f"with parallax {parallax:.2f} px.")

                curr_kf = potential_curr_kf
                R_best, t_best = R_cand, t_cand
                ids_best, p1_best, p2_best = ids_cand, p1_cand, p2_cand
                break
            else:
                print(f"  - Pair (KF {ref_kf.get_id()}, KF {potential_curr_kf.get_id()}) has insufficient parallax ({parallax:.2f} px).")

        if curr_kf is None:
            print("【Visual Init】: Failed to find a suitable pair in this window.")
            return False, None, None, None, None, None   

        # 三角化最优KF对的特征点
        points_3d_raw, mask_dpeth = self.sfm_processor.triangulate_points(p1_best, p2_best, R_best, t_best)

        if len(points_3d_raw) < 30:
            print(f"【Visual Init】: Triangulation resulted in too few valid points ({len(points_3d_raw)}).")
            return False, None, None, None, None, None

        p1_depth_ok = p1_best[mask_dpeth]
        p2_depth_ok = p2_best[mask_dpeth]

        final_points_3d, reprojection_mask = self.sfm_processor.filter_points_by_reprojection(
            points_3d_raw, p1_depth_ok, p2_depth_ok, R_best, t_best
        )

        if len(final_points_3d) < 30:
            print(f"【Visual Init】: Reprojection resulted in too few valid points ({len(final_points_3d)}).")
            return False, None, None, None, None, None

        intial_valid_ids = np.array(ids_best)[mask_dpeth]
        final_valid_ids = intial_valid_ids[reprojection_mask]

        print(f"【Visual Init】: Triangulation refined. Kept {len(final_points_3d)}/{len(points_3d_raw)} points.")

        # 加入地图
        self.landmarks.clear()
        for landmark_id, landmark_pt in zip(final_valid_ids, final_points_3d):
            self.landmarks[landmark_id] = landmark_pt

        # 设置curr_kf的位姿
        T_curr_ref = np.eye(4)
        T_curr_ref[:3, :3] = R_best
        T_curr_ref[:3, 3] = t_best.ravel()
        T_ref_curr = np.linalg.inv(T_curr_ref)
        curr_kf.set_global_pose(T_ref_curr)

        # 使用PnP计算其他KF位姿
        for kf in initial_keyframes:
            # 跳过参考KF和最新KF
            if kf.get_id() in [ref_kf.get_id(), curr_kf.get_id()]:
                continue

            success_pnp, pose = self.sfm_processor.track_with_pnp(self.landmarks, kf)
            # print(f"pose: {pose}")
            if success_pnp:
                kf.set_global_pose(pose)

        print(f"【Visual Init】: Success! Map has {len(self.landmarks)} landmarks.")
        
        return True, ref_kf, curr_kf, ids_best, p1_best, p2_best

    def process_package_data(self, new_kf):
        if len(self.keyframe_window) < 2:
            return

        last_kf = self.keyframe_window[-2]

        imu_factor_data = self.create_imu_factors(last_kf, new_kf)
        if not imu_factor_data:
            print(f"【Estimator】: No IMU factors between KF {last_kf.get_id()} and KF {new_kf.get_id()}.")
            return
