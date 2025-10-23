from symbol import factor
import gtsam
import numpy as np
import threading
import queue

from .backend import Backend
from datatype.keyframe import KeyFrame
from datatype.global_map import GlobalMap
from datatype.localmap import LocalMap
from datatype.landmark import Landmark, LandmarkStatus
from .imu_process import IMUProcessor
from .sfm_processor import SfMProcessor
from .viewer import Viewer3D
from utils.debug import Debugger
from .vio_initializer import VIOInitializer


def check_orthogonality(matrix, matrix_name):
    """检查3x3旋转矩阵的正交性"""
    R = matrix[:3, :3]
    # 计算 R' * R - I
    identity = np.eye(3)
    error_matrix = np.dot(R.T, R) - identity
    # 计算误差矩阵的范数，如果接近0，则说明是正交的
    error_norm = np.linalg.norm(error_matrix)
    
    is_orthogonal = np.allclose(error_norm, 0)
    
    if not is_orthogonal:
        print(f"🕵️‍ [Orthogonality Check] {matrix_name} FAILED! Error Norm: {error_norm:.6f}")
    else:
        print(f"✅ [Orthogonality Check] {matrix_name} PASSED. Error Norm: {error_norm:.6f}")
    
    return is_orthogonal

class Estimator(threading.Thread):
    """
    The central coordinator for the SLAM system. Runs as a consumer thread.
    """
    def __init__(self, config, input_queue, viewer_queue, global_central_map):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = input_queue
        self.local_map = LocalMap(config)

        self.imu_processor = IMUProcessor(config)
        self.imu_buffer = []

        self.backend = Backend(global_central_map, config)

        # 读取相机内参
        cam_intrinsics_raw = self.config.get('cam_intrinsics', np.eye(3).flatten().tolist())
        self.cam_intrinsics = np.asarray(cam_intrinsics_raw).reshape(3, 3)

        self.sfm_processor = SfMProcessor(self.cam_intrinsics)

        self.next_kf_id = 0

        # 初始化相关设置
        self.is_initialized = False

        self.init_window_size = self.config.get('init_window_size', 10)

        self.gravity_magnitude = self.config.get('gravity', 9.81)
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)
        self.T_cb = gtsam.Pose3(self.T_bc).inverse()

        # 可视化test
        self.viewer_queue = viewer_queue

        # 轨迹文件
        self.trajectory_file = None
        trajectory_output_path = self.config.get('trajectory_output_path', None) # self.config.get('trajectory_output_path', None)
        if trajectory_output_path:
            self.trajectory_file = Debugger.initialize_trajectory_file(trajectory_output_path)

        # Threading control
        self.is_running = False

    def start(self):
        self.is_running = True
        super().start()

    def shutdown(self):
        self.is_running = False
        if self.trajectory_file:
            self.trajectory_file.close()
            print("【Estimator】Trajectory file closed.")
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
                    visual_features = package['visual_features']
                    feature_ids = package['feature_ids']
                    image = package['image']

                    new_id = self.next_kf_id
                    new_kf = KeyFrame(new_id, timestamp)
                    new_kf.add_visual_features(visual_features, feature_ids)
                    new_kf.set_image(image)

                    self.next_kf_id += 1
                    self.local_map.add_keyframe(new_kf)

                    # 视觉惯性初始化
                    if not self.is_initialized:
                        active_keyframes = self.local_map.get_active_keyframes()
                        if len(active_keyframes) == self.init_window_size:
                            self.visual_inertial_initialization()
                        else:
                            print(f"【Init】: Collecting frames... {len(active_keyframes)}/{self.init_window_size}")
                    else:
                        # pass
                        self.process_new_keyframe(new_kf)

            except queue.Empty:
                continue
        
        print("【Estimator】thread has finished.")

    def create_imu_factors(self, kf_start, kf_end, latest_bias):
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

        imu_preintegration = self.imu_processor.pre_integration(measurements_with_ts, start_ts, end_ts, latest_bias)

        if imu_preintegration:
            return {
                'start_kf_timestamp': start_ts,
                'end_kf_timestamp': end_ts,
                'imu_measurements': measurements_with_ts,
                'start_kf_id': kf_start.get_id(),
                'end_kf_id': kf_end.get_id(),
                'imu_preintegration': imu_preintegration,
            }
        else:
            return None
        
    def triangulate_new_landmarks(self):
        new_triangulated_landmarks = {}
        keyframe_window = self.local_map.get_active_keyframes()
        # DEBUG
        suspect_lm_id = 14815
        # DEBUG
        for lm in self.local_map.get_candidate_landmarks():
            # DEBUG
            if lm.id == suspect_lm_id:
                print(f"🕵️‍ [Trace l{suspect_lm_id}]: Is a candidate. Checking for triangulation...")
            # DEBUG
            
            is_ready, first_kf, last_kf = lm.is_ready_for_triangulation(keyframe_window, min_parallax=50)

            # DEBUG
            if lm.id == suspect_lm_id and is_ready:
                print(f"🕵️‍ [Trace l{suspect_lm_id}]: PASSED triangulation check (ready). Using KF {first_kf.get_id()} and KF {last_kf.get_id()}.")
            # DEBUG
            
            # 检查是否能够晋升为正式landmark，通过观测的第一帧和最后一帧（last很可能是新加入的）
            if is_ready:
                pose1 = first_kf.get_global_pose()
                pose2 = last_kf.get_global_pose()
                
                if pose1 is None or pose2 is None:
                    continue

                T_2_1 = np.linalg.inv(pose2) @ pose1

                R, t = T_2_1[:3, :3], T_2_1[:3, 3].reshape(3, 1)

                pts1 = np.array([lm.get_observation(first_kf.get_id())])
                pts2 = np.array([lm.get_observation(last_kf.get_id())])

                points_3d_in_c1, mask = self.sfm_processor.triangulate_points(pts1, pts2, R, t)

                if len(points_3d_in_c1) > 0:
                    points_3d_world = (pose1[:3, :3] @ points_3d_in_c1.T + pose1[:3, 3].reshape(3, 1)).flatten()
                    # DEBUG
                    if lm.id == suspect_lm_id:
                        print(f"🕵️‍ [Trace l{suspect_lm_id}]: TRIANGULATED successfully to position {points_3d_world}.")
                    # DEBUG

                    # 检查是否满足视差角要求
                    is_healthy = self.local_map.check_landmark_health(lm.id, points_3d_world)
                    if is_healthy:
                        # 晋升为正式landmark
                        lm.set_triangulated(points_3d_world)
                        new_triangulated_landmarks[lm.id] = points_3d_world
                        # DEBUG
                        if lm.id == suspect_lm_id:
                            print(f"🕵️‍ [Trace l{suspect_lm_id}]: PASSED health check. Adding its factors...")
                        # DEBUG
                    
                    else:
                        # DEBUG
                        if lm.id == suspect_lm_id:
                            print(f"🕵️‍ [Trace l{suspect_lm_id}]: FAILED health check. Not adding its factors...")
                        # DEBUG
                        continue
                
                else:
                    if lm.id == suspect_lm_id:
                            print(f"🕵️‍ [Trace l{suspect_lm_id}]: FAILED multi-view validation after triangulation.")
    
        return new_triangulated_landmarks
            
    def visual_inertial_initialization(self):
        print("【Init】: Buffer is full. Starting initialization process.")

        initial_keyframes = self.local_map.get_active_keyframes()
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
            # 第一次跟踪时，直接使用imu_processor的初始bias
            imu_factors = self.create_imu_factors(kf_start, kf_end, None)
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
            for landmark_id, landmark_pt in zip(valid_ids, points_3d_world):
               if landmark_id in self.local_map.landmarks:
                    self.local_map.landmarks[landmark_id].set_triangulated(landmark_pt)
            print(f"【Init】: Re-triangulation complete. Final map has {len(self.local_map.landmarks)} landmarks.")
            print("【Init】: Alignment successful. Calling backend to build initial graph...")

            # 准备初始优化的变量
            initial_keyframes = self.local_map.get_active_keyframes()
            active_landmarks = self.local_map.get_active_landmarks()

            # 更新IMUProcessor的当前bias
            initial_bias_obj = gtsam.imuBias.ConstantBias(np.zeros(3), gyro_bias)
            self.imu_processor.update_bias(initial_bias_obj)
            
            # IMU因子
            initial_imu_factors = []
            for i in range(len(initial_keyframes) - 1):
                factor = self.create_imu_factors(initial_keyframes[i], initial_keyframes[i + 1], initial_bias_obj)
                if factor:
                    initial_imu_factors.append(factor)

            # 初始状态猜测
            initial_guesses = {}
            for i, kf in enumerate(initial_keyframes):
                initial_guesses[kf.get_id()] = (kf.get_global_pose(), velocities[i*3 : i*3+3], initial_bias_obj)

            # 进行初始优化
            success = self.backend.optimize(
                keyframe_window=initial_keyframes,
                imu_factors=initial_imu_factors, 
                active_landmarks=active_landmarks, 
                initial_state_guess=initial_guesses
            )

            if success:
                self.is_initialized = True
                print("【Init】: Initialization successful and initial graph optimized.")

                # viewer可视化
                if self.viewer_queue:
                    print("【Tracking】: Sending tracking result to viewer queue...")

                    # 从 local_map 中获取最新的、优化后的位姿和路标点数据
                    active_kfs = self.local_map.get_active_keyframes()
                    poses = {kf.get_id(): kf.get_global_pose() for kf in active_kfs if kf.get_global_pose() is not None}
                    
                    # 【核心修正】调用 LocalMap 的辅助函数来获取纯粹的位置字典
                    active_landmarks_objects = self.local_map.get_active_landmarks()

                    landmarks_positions = {lm_id: lm_obj.get_position() for lm_id, lm_obj in active_landmarks_objects.items() if lm_obj.get_position() is not None}

                    vis_data = {
                        'landmarks': landmarks_positions,
                        'poses': poses
                    }
                    
                    # 打印一些信息以供调试
                    print(f"【Viewer】: Sending {len(poses)} poses and {len(landmarks_positions)} landmarks to viewer.")

                    try:
                        self.viewer_queue.put_nowait(vis_data)
                    except queue.Full:
                        print("【Estimator】: Viewer queue is full, skipping visualization data.")
                # viewer可视化
            else:
                print("【Init】: Backend optimization failed during initialization.")
            
        else:
            print("【Init】: V-I Alignment failed.")

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

            # 保证一定的视差，这里是一个非常敏感的参数，每次代码改动都可能需要重新调整这个参数
            if parallax > 70:
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

        # 转换为字典形式，方便后续PnP使用
        sfm_landmarks = {lm_id: pt for lm_id, pt in zip(final_valid_ids, final_points_3d)}

        self.local_map.landmarks.clear()
    
        all_feature_maps = {}
        for kf in initial_keyframes:
            all_feature_maps[kf.get_id()] = {fid: feat for fid, feat in zip(kf.get_visual_feature_ids(), kf.get_visual_features())}

        for lm_id in sfm_landmarks.keys():
            # 找到第一个观测到该路标点的KF来创建Landmark对象
            first_obs_kf = None
            for kf in initial_keyframes:
                if lm_id in all_feature_maps[kf.get_id()]:
                    first_obs_kf = kf
                    break

            if first_obs_kf:
                pt_2d = all_feature_maps[first_obs_kf.get_id()][lm_id]
                new_lm = Landmark(lm_id, first_obs_kf.get_id(), pt_2d)

                # 添加这个路标点在其他KF的观测
                for kf in initial_keyframes:
                    if kf.get_id() != first_obs_kf.get_id() and lm_id in all_feature_maps[kf.get_id()]:
                            new_lm.add_observation(kf.get_id(), all_feature_maps[kf.get_id()][lm_id])

                self.local_map.landmarks[lm_id] = new_lm

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

            success_pnp, pose = self.sfm_processor.track_with_pnp(sfm_landmarks, kf)
            # print(f"pose: {pose}")
            if success_pnp:
                kf.set_global_pose(pose)

        print(f"【Visual Init】: Success! Map has {len(sfm_landmarks)} landmarks.")
        
        return True, ref_kf, curr_kf, ids_best, p1_best, p2_best

    def process_new_keyframe(self, new_kf):
        active_kfs = self.local_map.get_active_keyframes()
        if len(active_kfs) < 2:
            return

        # 获取上一帧的位姿、速度、偏置
        last_kf = active_kfs[-2]
        last_pose_mat, last_vel, last_bias = last_kf.get_global_pose(), last_kf.get_velocity(), last_kf.get_bias()
        # last_pose_gtsam = gtsam.Pose3(last_pose_mat).compose(self.T_cb)
        
        # 净化Pose矩阵的代码
        last_R_mat = last_pose_mat[:3, :3]
        last_t_vec = last_pose_mat[:3, 3]
        last_rot = gtsam.Rot3(last_R_mat)
        last_pos = gtsam.Point3(last_t_vec)
        last_pose_gtsam_wc = gtsam.Pose3(last_rot, last_pos)

        last_pose_gtsam_wb = last_pose_gtsam_wc.compose(self.T_cb)

        # 创建上一帧到当前帧的IMU因子
        imu_factor_data = self.create_imu_factors(last_kf, new_kf, last_bias)
        if not imu_factor_data:
            print(f"【Estimator】: No IMU factors between KF {last_kf.get_id()} and KF {new_kf.get_id()}.")
            return

        # 使用当前帧的预积分来预测当前帧状态
        pim = imu_factor_data['imu_preintegration']
        predicted_nav_state = pim.predict(gtsam.NavState(last_pose_gtsam_wb, last_vel), last_bias)

        predicted_T_wb = predicted_nav_state.pose()
        predicted_T_wc = predicted_T_wb.compose(gtsam.Pose3(self.T_bc))
        predicted_vel = predicted_nav_state.velocity()

        # 创建临时预测
        new_kf.set_global_pose(predicted_T_wc.matrix())
        new_kf.set_velocity(predicted_vel)
        new_kf.set_bias(last_bias)

        # 进行特征点延迟三角化
        #【错误点修正】不应该在优化前，使用一个纯粹靠IMU预测的、未经视觉信息约束的位姿来进行三角化
        # new_landmarks = self.triangulate_new_landmarks()
        # if new_landmarks:
        #     print(f"【Estimator】: Triangulated {len(new_landmarks)} new landmarks.")

        # 准备优化所需的所有数据
        keyframe_window = self.local_map.get_active_keyframes()
        active_landmarks = self.local_map.get_active_landmarks()

        # 创建所有IMU因子
        imu_factors = []
        for i in range(len(keyframe_window) - 1):
            kf_start = keyframe_window[i]
            kf_end = keyframe_window[i + 1]
            
            # 这里使用每段积分开头的KF的bias作为该段积分的bias
            start_kf_bias = kf_start.get_bias()
            # 处理第一次跟踪时，旧帧可能没有bias的情况
            if start_kf_bias is None:
                start_kf_bias = self.imu_processor.current_bias # 使用IMUProcessor的当前bias作为备用
            
            imu_factor = self.create_imu_factors(kf_start, kf_end, start_kf_bias)
            if imu_factor:
                imu_factors.append(imu_factor)

        # 设置初始状态猜测
        initial_guesses = {}
        for kf in keyframe_window:
            if kf.get_id() == new_kf.get_id():
                initial_guesses[kf.get_id()] = (predicted_T_wc.matrix(), predicted_vel, last_bias)
            else:
                initial_guesses[kf.get_id()] = (kf.get_global_pose(), kf.get_velocity(), kf.get_bias())

        # 将预测结果作为初始估计值以及重投影约束、IMU约束送入后端
        success = self.backend.optimize(
            keyframe_window=keyframe_window,
            imu_factors=imu_factors,
            active_landmarks=active_landmarks,
            initial_state_guess=initial_guesses
        )

        if success:
            print(f"【Estimator】: Optimization successful for KF {new_kf.get_id()}.")
            
            #【时机修正】在优化运行成功之后，KF的位姿已经更新，此时进行三角化更准确
            new_landmarks = self.triangulate_new_landmarks()
            if new_landmarks:
                print(f"【Estimator】: Triangulated {len(new_landmarks)} new landmarks.")

            # 使用优化后的最后一帧bias作为最新的bias
            latest_kf = self.local_map.get_active_keyframes()[-1]
            self.imu_processor.update_bias(latest_kf.get_bias())

            # viewer可视化
            if self.viewer_queue:
                print("【Tracking】: Sending tracking result to viewer queue...")

                # 从 local_map 中获取最新的、优化后的位姿和路标点数据
                active_kfs = self.local_map.get_active_keyframes()
                poses = {kf.get_id(): kf.get_global_pose() for kf in active_kfs if kf.get_global_pose() is not None}
                
                # 【核心修正】调用 LocalMap 的辅助函数来获取纯粹的位置字典
                active_landmarks_objects = self.local_map.get_active_landmarks()

                landmarks_positions = {lm_id: lm_obj.get_position() for lm_id, lm_obj in active_landmarks_objects.items() if lm_obj.get_position() is not None}

                vis_data = {
                    'landmarks': landmarks_positions,
                    'poses': poses
                }
                
                # 打印一些信息以供调试
                print(f"【Viewer】: Sending {len(poses)} poses and {len(landmarks_positions)} landmarks to viewer.")

                try:
                    self.viewer_queue.put_nowait(vis_data)
                except queue.Full:
                    print("【Estimator】: Viewer queue is full, skipping visualization data.")
            # viewer可视化
            
        else:
            print(f"【Estimator】: Optimization failed for KF {new_kf.get_id()}.")
