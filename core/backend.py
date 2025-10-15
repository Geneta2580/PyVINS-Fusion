import queue
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B, L

class Backend:
    def __init__(self, global_central_map, config, imu_processor):
        self.global_central_map = global_central_map
        self.config = config

        # 使用 iSAM2 作为优化器
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1) 
        parameters.relinearizeSkip = 1
        self.isam2 = gtsam.ISAM2(parameters)
        
        # 状态与id管理
        self.kf_id_to_gtsam_id = {}
        self.landmark_id_to_gtsam_id = {}
        self.next_gtsam_kf_id = 0
        
        # 获取相机内、外参
        cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)
        self.K = gtsam.Cal3_S2(cam_intrinsics[0, 0], cam_intrinsics[1, 1], 0, 
                               cam_intrinsics[0, 2], cam_intrinsics[1, 2])

        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)
        self.body_T_cam = gtsam.Pose3(self.T_bc)
        self.cam_T_body = self.body_T_cam.inverse()

        # 存储最新的优化后的偏置，用于IMU预积分
        self.latest_bias = gtsam.imuBias.ConstantBias()

    # 关键帧id映射到图的id
    def _get_kf_gtsam_id(self, kf_id):
        if kf_id not in self.kf_id_to_gtsam_id:
            self.kf_id_to_gtsam_id[kf_id] = self.next_gtsam_kf_id
            self.next_gtsam_kf_id += 1
        return self.kf_id_to_gtsam_id[kf_id]

    # 路标点id映射到图的id
    def _get_lm_gtsam_id(self, lm_id):
        if lm_id not in self.landmark_id_to_gtsam_id:
            self.landmark_id_to_gtsam_id[lm_id] = lm_id
        return self.landmark_id_to_gtsam_id[lm_id]

    def get_latest_optimized_state(self):
        if self.next_gtsam_kf_id == 0:
            return None, None, None
        
        latest_gtsam_id = self.next_gtsam_kf_id - 1
        result = self.isam2.calculateEstimate()

        try:
            pose = result.atPose3(X(latest_gtsam_id))
            velocity = result.atVector(V(latest_gtsam_id))
            bias = result.atConstantBias(B(latest_gtsam_id))
            return pose, velocity, bias
        except Exception as e:
            print(f"[Error][Backend] Failed to retrieve latest state for gtsam_id {latest_gtsam_id}: {e}")
            return None, None, None

    def update_estimator_map(self, keyframe_window, landmarks):
        print("【Backend】: Syncing optimized results back to Estimator...")
        optimized_results = self.isam2.calculateEstimate()

        # 更新关键帧位姿
        for kf in keyframe_window:
           # 获取待更新关键帧的gtsam_id
            gtsam_id = self.kf_id_to_gtsam_id.get(kf.get_id())
            if gtsam_id is not None and optimized_results.exists(X(gtsam_id)):
                
                # 从优化结果中获取最新的IMU位姿 T_w_b
                pose_w_b = optimized_results.atPose3(X(gtsam_id))
                
                # 转换回相机位姿 T_w_c 并更新
                pose_w_c = pose_w_b.compose(self.body_T_cam)
                kf.set_global_pose(pose_w_c.matrix())

        # 更新路标点坐标
        for lm_id in landmarks.keys():
            gtsam_id = self.landmark_id_to_gtsam_id.get(lm_id)
            if gtsam_id is not None and optimized_results.exists(L(gtsam_id)):
                landmarks[lm_id] = optimized_results.atPoint3(L(gtsam_id))

    def initialize_optimize(self, initial_keyframes, initial_imu_factors, initial_landmarks, initial_velocities, initial_bias):
        print("【Backend】: Initializing optimize...")

        graph = gtsam.NonlinearFactorGraph()
        estimates = gtsam.Values()

        for i, kf in enumerate(initial_keyframes):
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            
            # 从初始化结果中获取位姿、速度和偏置
            T_wc = gtsam.Pose3(kf.get_global_pose())
            T_wb = T_wc.compose(self.cam_T_body)
            # initial_velocities 是一个扁平化的数组，每3个元素是一个速度向量
            velocity = initial_velocities[i*3 : i*3+3]
            
            # 只有第一个关键帧的偏置来自初始化器，后续的偏置在因子图中传递
            bias = initial_bias if i == 0 else self.latest_bias

            estimates.insert(X(kf_gtsam_id), T_wb)
            estimates.insert(V(kf_gtsam_id), velocity)
            estimates.insert(B(kf_gtsam_id), bias)

            # 为第一帧添加强先验
            if kf_gtsam_id == 0:
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*3 + [1e-4]*3))
                prior_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4] * 3))
                prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2]*3 + [1e-3]*3))
                graph.add(gtsam.PriorFactorPose3(X(0), T_wb, prior_pose_noise))
                graph.add(gtsam.PriorFactorVector(V(0), velocity, prior_vel_noise))
                graph.add(gtsam.PriorFactorConstantBias(B(0), bias, prior_bias_noise))

        # 添加所有初始IMU因子
        for factor_data in initial_imu_factors:
            start_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['start_kf_timestamp'])
            end_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['end_kf_timestamp'])
            gtsam_id1 = self._get_kf_gtsam_id(start_kf.get_id())
            gtsam_id2 = self._get_kf_gtsam_id(end_kf.get_id())
            pim = factor_data['imu_preintegration']
            graph.add(gtsam.CombinedImuFactor(X(gtsam_id1), V(gtsam_id1), X(gtsam_id2), V(gtsam_id2), B(gtsam_id1), B(gtsam_id2), pim))

        # 添加所有初始路标点变量和视觉因子
        visual_factor_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)
        for lm_id, lm_3d_pos in initial_landmarks.items():
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            estimates.insert(L(lm_gtsam_id), lm_3d_pos)

        for kf in initial_keyframes:
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                # 只处理本次优化中新添加的landmark
                if lm_id in initial_landmarks:
                    lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
                    factor = gtsam.GenericProjectionFactorCal3_S2(pt_2d, visual_factor_noise, X(kf_gtsam_id), L(lm_gtsam_id), self.K, body_P_sensor=self.body_T_cam)
                    graph.add(factor)
        
        # 执行iSAM2的第一次更新（批量模式）
        print(f"【Backend】: Initializing iSAM2 with {graph.size()} new factors and {estimates.size()} new values...")
        self.isam2.update(graph, estimates)
        for _ in range(2): self.isam2.update()
        
        # 更新最新bias
        self.get_latest_optimized_state()
        print("【Backend】: Initial graph optimization complete.")


    def optimize_incremental(self, last_keyframe, new_keyframe, new_imu_factors, new_landmarks, initial_state_guess):

        new_graph = gtsam.NonlinearFactorGraph()
        new_estimates = gtsam.Values()

        # 添加新关键帧的状态变量，使用IMU预测值作为初始估计
        kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())
        T_wb_guess, vel_guess, bias_guess = initial_state_guess

        new_estimates.insert(X(kf_gtsam_id), T_wb_guess)
        new_estimates.insert(V(kf_gtsam_id), vel_guess)
        new_estimates.insert(B(kf_gtsam_id), bias_guess)

        # 添加IMU因子
        last_kf_gtsam_id = self._get_kf_gtsam_id(last_keyframe.get_id())
        pim = new_imu_factors['imu_preintegration']
        imu_factor = gtsam.CombinedImuFactor(
            X(last_kf_gtsam_id), V(last_kf_gtsam_id), X(kf_gtsam_id), V(kf_gtsam_id),
            B(last_kf_gtsam_id), B(kf_gtsam_id), pim)
        new_graph.add(imu_factor)

        visual_factor_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)
        for lm_id, lm_3d_pos in new_landmarks.items():
            if not self.isam2.getLinearizationPoint().exists(L(lm_id)): # 检查点是否已在图中
                lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
                new_estimates.insert(L(lm_gtsam_id), lm_3d_pos)
        
        # 为新关键帧添加重投影因子
        for lm_id, pt_2d in zip(new_keyframe.get_visual_feature_ids(), new_keyframe.get_visual_features()):
            if self.isam2.getLinearizationPoint().exists(L(lm_id)) or lm_id in new_landmarks:
                lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
                factor = gtsam.GenericProjectionFactorCal3_S2(pt_2d, visual_factor_noise, X(kf_gtsam_id), L(lm_gtsam_id), self.K, body_P_sensor=self.body_T_cam)
                new_graph.add(factor)

        # 行iSAM2增量更新
        print(f"【Backend】: Updating iSAM2 ({new_graph.size()} new factors, {new_estimates.size()} new variables)...")
        self.isam2.update(new_graph, new_estimates)
        for _ in range(2): self.isam2.update()

        # 更新最新bias
        self.get_latest_optimized_state()
        print("【Backend】: Incremental optimization complete.")