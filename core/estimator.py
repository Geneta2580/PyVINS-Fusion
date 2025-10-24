import gtsam
import numpy as np
import threading
import queue
import time

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

        self.backend = Backend(global_central_map, config, self.imu_processor)

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
                    timestamp = package['timestamp']
                    visual_features = package['visual_features']
                    feature_ids = package['feature_ids']
                    image = package['image']

                    new_id = self.next_kf_id
                    new_kf = KeyFrame(new_id, timestamp)
                    new_kf.add_visual_features(visual_features, feature_ids)
                    new_kf.set_image(image)

                    self.next_kf_id += 1
                    stale_lm_ids = self.local_map.add_keyframe(new_kf)

                    # if stale_lm_ids:
                    #     self.backend.remove_stale_landmarks(stale_lm_ids)

                    active_keyframes = self.local_map.get_active_keyframes()

                    if not self.is_initialized:
                        if len(active_keyframes) == self.init_window_size:
                            self.visual_inertial_initialization()
                        
                        else:
                            print(f"【Init】: Collecting frames... {len(active_keyframes)}/{self.init_window_size}")
                    else:
                        # pass
                        self.process_package_data(new_kf)

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

    def triangulate_new_landmarks(self, new_kf):
        newly_triangulated_for_backend = {}
        keyframe_window = self.local_map.get_active_keyframes()
        # DEBUG
        suspect_lm_id = 14815
        # DEBUG
        for lm in self.local_map.get_candidate_landmarks():
            # DEBUG
            if lm.id == suspect_lm_id:
                print(f"🕵️‍ [Trace l{suspect_lm_id}]: Is a candidate. Checking for triangulation...")
            # DEBUG
            
            is_ready, first_kf, last_kf = lm.is_ready_for_triangulation(keyframe_window, min_parallax=30)

            # DEBUG
            if lm.id == suspect_lm_id and is_ready:
                print(f"🕵️‍ [Trace l{suspect_lm_id}]: PASSED triangulation check (ready). Using KF {first_kf.get_id()} and KF {last_kf.get_id()}.")
            # DEBUG
            
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

                    is_healthy = self.local_map.check_landmark_health(lm.id, points_3d_world)
                    if is_healthy:
                        lm.set_triangulated(points_3d_world)
                        newly_triangulated_for_backend[lm.id] = points_3d_world
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
    
        return newly_triangulated_for_backend
            
    
    # 审计地图，移除所有变得不健康的坏点
    def audit_map_after_optimization(self):
        landmarks_to_remove = []
        # 遍历所有已三角化的路标点
        for lm_id in self.local_map.get_active_landmarks().keys():
            # 使用同一个健康检查函数进行“年检”
            # is_healthy = self.local_map.check_landmark_health(lm_id)
            is_depth_ok = self.local_map.check_landmark_depth(lm_id)
            if not is_depth_ok:
                landmarks_to_remove.append(lm_id)
        
        if landmarks_to_remove:
            print(f"【Audit】: Removing {len(landmarks_to_remove)} landmarks that became unhealthy after optimization: {landmarks_to_remove}")
            # 从LocalMap中删除
            for lm_id in landmarks_to_remove:
                if lm_id in self.local_map.landmarks:
                    del self.local_map.landmarks[lm_id]

            self.backend.remove_stale_landmarks(landmarks_to_remove)
    
    # 零速检查
    def is_stationary(self, imu_measurements_between_kfs):
        if len(imu_measurements_between_kfs) < 10: # 至少需要一些样本
            return False

        # 提取所有的加速度和角速度读数
        accel_list = [m[1].accel.astype(np.float64) for m in imu_measurements_between_kfs]
        gyro_list = [m[1].gyro.astype(np.float64) for m in imu_measurements_between_kfs]

        try:
            accels = np.array(accel_list, dtype=np.float64)         
            gyros = np.array(gyro_list, dtype=np.float64)
        except ValueError:
            print("【Stationary Check】: Failed to convert IMU measurements to numpy arrays.")
            return False

        # 计算加速度和角速度在每个轴上的标准差
        accel_std = np.std(accels, axis=0)
        gyro_std = np.std(gyros, axis=0)

        # 从config中读取阈值
        accel_std_threshold = self.config.get('stationary_accel_std_threshold', 0.05) # m/s^2
        gyro_std_threshold = self.config.get('stationary_gyro_std_threshold', 0.05) # rad/s

        # 如果所有轴的波动都小于阈值，则认为是静止
        is_still = np.all(accel_std < accel_std_threshold) and np.all(gyro_std < gyro_std_threshold)

        if is_still:
            print("【Stationary Check】: System is stationary.")
            
        return is_still


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
            for landmark_id, landmark_pt in zip(valid_ids, points_3d_world):
               if landmark_id in self.local_map.landmarks:
                    self.local_map.landmarks[landmark_id].set_triangulated(landmark_pt)
            print(f"【Init】: Re-triangulation complete. Final map has {len(self.local_map.landmarks)} landmarks.")
            print("【Init】: Alignment successful. Calling backend to build initial graph...")

            # 更新IMU偏置
            initial_bias_obj = gtsam.imuBias.ConstantBias(np.zeros(3), gyro_bias)
            self.imu_processor.update_bias(initial_bias_obj)
            
            # test
            poses = {kf.get_id(): kf.get_global_pose() for kf in initial_keyframes if kf.get_global_pose() is not None}
            for kf_id, pose in poses.items():
                print(f"【Init】: Before optimization. kf_id: {kf_id}, pose: {pose[:3, 3]}")
            # test

            # 进行初始优化
            self.backend.initialize_optimize(
                self.local_map.get_active_keyframes(),
                initial_imu_factors, 
                self.local_map.get_active_landmarks(), 
                velocities, initial_bias_obj
            )

            # 初始优化结束，同步后端结果到Estimator
            self.backend.update_estimator_map(
                self.local_map.get_active_keyframes(),
                self.local_map.landmarks
            )
            self.is_initialized = True

            # 记录初始优化轨迹
            if self.trajectory_file:
                print("【Estimator】正在记录初始优化轨迹...")
                # 按时间戳排序以确保轨迹顺序正确
                sorted_kfs = sorted(self.local_map.get_active_keyframes(), key=lambda kf: kf.get_timestamp())
                for kf in sorted_kfs:
                    Debugger.log_trajectory_tum(self.trajectory_file, kf) # 调用静态方法
                self.trajectory_file.flush() # 确保数据立即写入磁盘
            # 记录初始优化轨迹
            
            #  viewer可视化
            if self.viewer_queue:
                print("【Init】: Sending initialization result to viewer queue...")

                # 从 local_map 中获取最新的、优化后的位姿和路标点数据
                active_kfs = self.local_map.get_active_keyframes()
                poses = {kf.get_id(): kf.get_global_pose() for kf in active_kfs if kf.get_global_pose() is not None}
                
                # 调用 LocalMap 的辅助函数来获取纯粹的位置字典
                landmarks_positions = self.local_map.get_active_landmarks()

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

    def process_package_data(self, new_kf):
        active_kfs = self.local_map.get_active_keyframes()
        if len(active_kfs) < 2:
            return

        last_kf = active_kfs[-2]

        # 创建上一帧到当前帧的IMU因子
        start_time = time.time()
        imu_factor_data = self.create_imu_factors(last_kf, new_kf)
        end_time = time.time()
        print(f"【Estimator Timer】: IMU Factor Creation took {(end_time - start_time) * 1000:.3f} ms.")
        if not imu_factor_data:
            print(f"【Estimator】: No IMU factors between KF {last_kf.get_id()} and KF {new_kf.get_id()}.")
            return

        is_currently_stationary = self.is_stationary(imu_factor_data['imu_measurements']) # 零速检查

        # 从后端获取最新的优化结果
        last_pose, last_vel, last_bias = self.backend.get_latest_optimized_state()
        if last_pose is None:
            print(f"[Warning] Could not retrieve last state from backend. Skipping KF {new_kf.get_id()}.")
            return

        # 使用当前帧的预积分来预测当前帧状态
        pim = imu_factor_data['imu_preintegration']
        predicted_nav_state = pim.predict(gtsam.NavState(last_pose, last_vel), last_bias)

        predicted_T_wb = predicted_nav_state.pose()
        predicted_T_wc = predicted_T_wb.compose(gtsam.Pose3(self.T_bc))
        predicted_vel = predicted_nav_state.velocity()

        # 设置临时预测位姿
        new_kf.set_global_pose(predicted_T_wc.matrix())

        # 进行新特征点三角化
        new_landmarks = self.triangulate_new_landmarks(new_kf)
        if new_landmarks:
            print(f"【Tracking】: Triangulated {len(new_landmarks)} new landmarks.")
            print(f"【Tracking】: New landmarks: {new_landmarks.keys()}")

        # DEBUG
        suspect_lm_id = 14815
        # DEBUG
        
        # 为后端准备重投影因子
        visual_factors_to_add = []
        active_kf_ids = {kf.get_id() for kf in self.local_map.get_active_keyframes()}
        for lm_id in new_landmarks.keys():
            # DEBUG
            if lm_id == suspect_lm_id:
                print(f"🕵️‍ [Trace l{suspect_lm_id}]: Is newly triangulated. Preparing its factors...")
            # DEBUG
            lm = self.local_map.landmarks.get(lm_id)
            if lm:
                # DEBUG
                if lm_id == suspect_lm_id:
                    print(f"🕵️‍ [Trace l{suspect_lm_id}]: As new landmark. PASSED health check. Adding its factors...")
                # DEBUG
                for obs_kf_id, obs_pt_2d in lm.observations.items():
                    # 只添加活跃窗口内关键帧的观测
                    if obs_kf_id in active_kf_ids:
                        # 指令格式: (关键帧ID, 路标点ID, 2D观测坐标)
                        visual_factors_to_add.append((obs_kf_id, lm_id, obs_pt_2d))
                        print(f"🕵️‍ [Trace l{lm_id}]: OBSERVED by new KF {obs_kf_id}. observation point: {obs_pt_2d}")
                        # if lm_id == suspect_lm_id:
                        #     print(f"🕵️‍ [Trace l{suspect_lm_id}]: OBSERVED by new KF {obs_kf_id}. observation point: {obs_pt_2d}")
                            # print(f"🕵️‍ [Trace l{suspect_lm_id}]: OBSERVED by new KF {obs_kf_id}. observation point: {obs_pt_2d}")

        # 添加旧点重投影因子 (不在新三角化列表里)
        for lm_id, pt_2d in zip(new_kf.get_visual_feature_ids(), new_kf.get_visual_features()):
            if lm_id not in new_landmarks:
                # 必须是活跃点 (没有被剔除) 且已经三角化
                if lm_id in self.local_map.landmarks and self.local_map.landmarks[lm_id].status == LandmarkStatus.TRIANGULATED:
                    # DEBUG
                    if lm_id == suspect_lm_id:
                        print(f"🕵️‍ [Trace l{suspect_lm_id}]: As old landmark. PASSED health check. Adding its factors...")
                        # print(f"🕵️‍ [Trace l{suspect_lm_id}]: lm}")
                    # DEBUG
                    visual_factors_to_add.append((new_kf.get_id(), lm_id, pt_2d))

        print(f"【Debug】: Newly triangulated landmarks count: {len(new_landmarks)}")
        print(f"【Debug】: Factor instructions generated: {len(visual_factors_to_add)}")

        # 将预测结果作为初始估计值以及重投影约束、IMU约束送入后端
        start_time = time.time()
        self.backend.optimize_incremental(
            last_keyframe=last_kf,
            new_keyframe=new_kf,
            new_imu_factors=imu_factor_data,
            new_landmarks=new_landmarks,
            new_visual_factors=visual_factors_to_add,
            initial_state_guess=(predicted_T_wb, predicted_vel, last_bias),
            is_stationary=is_currently_stationary,
        )
        end_time = time.time()
        print(f"【Estimator Timer】: Backend Incremental Optimization took {(end_time - start_time) * 1000:.3f} ms.")

        # 优化结束，同步后端结果到Estimator
        self.backend.update_estimator_map(active_kfs, self.local_map.landmarks)
        
        # 审计地图，移除所有变得不健康的“坏苹果”
        start_time = time.time()
        # self.audit_map_after_optimization()
        end_time = time.time()
        print(f"【Estimator Timer】: Map Audit took {(end_time - start_time) * 1000:.3f} ms.")
        
        # 更新预积分器的零偏
        _, _, latest_bias = self.backend.get_latest_optimized_state()
        if latest_bias:
            self.imu_processor.update_bias(latest_bias)

        # 记录优化轨迹
        Debugger.log_trajectory_tum(self.trajectory_file, new_kf) # 调用静态方法
        if self.trajectory_file:
            self.trajectory_file.flush() # 确保数据立即写入磁盘
        # 记录优化轨迹
        
        # viewer可视化
        if self.viewer_queue:
            print("【Tracking】: Sending tracking result to viewer queue...")

            # 从 local_map 中获取最新的、优化后的位姿和路标点数据
            active_kfs = self.local_map.get_active_keyframes()
            poses = {kf.get_id(): kf.get_global_pose() for kf in active_kfs if kf.get_global_pose() is not None}
            
            # 【核心修正】调用 LocalMap 的辅助函数来获取纯粹的位置字典
            landmarks_positions = self.local_map.get_active_landmarks()

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
