import gtsam
import numpy as np
import threading
import queue

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
    def __init__(self, config, input_queue, global_central_map):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = input_queue
        self.global_map = global_central_map

        self.imu_processor = IMUProcessor(config)
        self.imu_buffer = []

        # 读取相机内参
        cam_intrinsics_raw = self.config.get('cam_intrinsics', np.eye(3).flatten().tolist())
        self.cam_intrinsics = np.asarray(cam_intrinsics_raw).reshape(3, 3)

        self.sfm_processor = SfMProcessor(self.cam_intrinsics)

        self.next_kf_id = 0
        self.keyframe_window = {}
        self.landmarks = {}

        # 初始化相关设置
        self.is_initialized = False
        self.init_kf_buffer = []
        self.init_imu_factors_buffer = []
        self.init_window_size = self.config.get('init_window_size', 10)

        # 可视化test
        self.viewer = Viewer3D()

        # Threading control
        self.is_running = False
        self.thread = threading.Thread(target=self.run, daemon=True)

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
                    print(f"【Estimator】: Received KeyFrame at timestamp {timestamp:.4f}")

                    if not self.is_initialized:
                        self.is_initialized = self.visual_inertial_initialization(package)
                    else:
                        pass
                        # self.process_package_data(package)

            except queue.Empty:
                continue
        
        print("【Estimator】thread has finished.")

        
    def visual_inertial_initialization(self, kf_package):
        timestamp = kf_package['timestamp']
        visual_features = kf_package['visual_features']
        feature_ids = kf_package['feature_ids']

        new_id = self.next_kf_id
        new_kf = KeyFrame(new_id, timestamp)
        new_kf.add_visual_features(visual_features, feature_ids)
        self.init_kf_buffer.append(new_kf)
        self.next_kf_id += 1

        if len(self.init_kf_buffer) < self.init_window_size:
            print(f"【Init】: Collecting frames... {len(self.init_kf_buffer)}/{self.init_window_size}")
            return

        print("【Init】: Buffer is full. Starting initialization process.")

        sfm_success = self.visual_initialization(self.init_kf_buffer)

        if sfm_success:
            if self.viewer:
                # 收集所有已计算的位姿
                poses = {kf.get_id(): kf.get_global_pose() for kf in self.keyframes.values() if kf.get_global_pose() is not None}
                self.viewer.update(self.landmarks, poses)

        return sfm_success
    
    
    def visual_initialization(self, initial_keyframes):
        print("【Visual Init】: Searching for the best keyframe pair...")
        curr_kf = initial_keyframes[-1]  # 最新KF
        ref_kf = None # 参考最优KF

        R, t, inlier_ids, pts1_inliers, pts2_inliers = [None] * 5

        # 从最新的KF开始，向前找到最优的KF对
        for i in range(len(initial_keyframes) - 2, -1, -1):
            potential_ref_kf = initial_keyframes[i]

            success, ids_cand, p1_cand, p2_cand, R_cand, t_cand = \
            self.sfm_processor.epipolar_compute(potential_ref_kf, curr_kf)

            if success:
                ref_kf = potential_ref_kf
                R, t, inlier_ids, pts1_inliers, pts2_inliers = R_cand, t_cand, ids_cand, p1_cand, p2_cand
                
                print(f"【Visual Init】: Found a good pair! (KF {ref_kf.get_id()}, KF {curr_kf.get_id()}).")

                # 三角化最优KF对的特征点
                points_3d, mask = self.sfm_processor.triangulate_points(pts1_inliers, pts2_inliers, R, t)
                if len(points_3d) < 30:
                    print(f"【Visual Init】: Triangulation resulted in too few valid points ({len(points_3d)}).")
                    continue

                # 最终的内点id
                valid_ids = np.array(inlier_ids)[mask]        
                
                # 加入地图
                for landmark_id, landmark_pt in zip(valid_ids, points_3d):
                    self.landmarks[landmark_id] = landmark_pt
                
                # 设置参考KF的位姿为原点
                ref_kf.set_global_pose(np.eye(4))

                T_curr_ref = np.eye(4)
                T_curr_ref[:3, :3] = R
                T_curr_ref[:3, 3] = t.ravel()
                T_ref_curr = np.linalg.inv(T_curr_ref)
                curr_kf.set_global_pose(T_ref_curr)

                # 加入到全局KF缓存
                self.keyframes[ref_kf.get_id()] = ref_kf
                self.keyframes[curr_kf.get_id()] = curr_kf

                # 使用PnP计算其他KF位姿
                for kf in initial_keyframes:
                    # 跳过参考KF和最新KF
                    if kf.get_id() == ref_kf.get_id() or kf.get_id() == curr_kf.get_id():
                        continue

                    success_pnp, pose = self.sfm_processor.track_with_pnp(self.landmarks, kf)
                    if success_pnp:
                        kf.set_global_pose(pose)
                        self.keyframes[kf.get_id()] = kf

                print(f"【Visual Init】: Success! Map has {len(self.landmarks)} landmarks.")
                return True

        print("【Visual Init】: Failed to initialize in this window.")
        return False


    def process_package_data(self, kf_package):
        timestamp = kf_package['timestamp']
        visual_features = kf_package['visual_features']
        feature_ids = kf_package['feature_ids']

        # 创建新KF
        new_id = self.next_kf_id
        new_kf = KeyFrame(new_id, timestamp)

        # 第一个KF
        if not self.keyframe_window:
            self.keyframe_window[timestamp] = new_id
            new_kf.add_visual_features(visual_features, feature_ids)
            self.next_kf_id += 1
            return

        prev_kf_id = max(self.keyframe_window.keys())
        prev_kf = self.keyframe_window[prev_kf_id]

        new_kf.add_visual_features(visual_features, feature_ids)
        self.keyframe_window[new_id] = new_kf
        self.next_kf_id += 1