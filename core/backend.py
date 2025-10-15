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

        # 存储最新的优化后的偏置，用于IMU预积分
        self.latest_bias = gtsam.imuBias.ConstantBias()

    def _get_kf_gtsam_id(self, kf_id):
        if kf_id not in self.kf_id_to_gtsam_id:
            self.kf_id_to_gtsam_id[kf_id] = self.next_gtsam_kf_id
            self.next_gtsam_kf_id += 1
        return self.kf_id_to_gtsam_id[kf_id]

    def _get_lm_gtsam_id(self, lm_id):
        if lm_id not in self.landmark_id_to_gtsam_id:
            self.landmark_id_to_gtsam_id[lm_id] = lm_id
        return self.landmark_id_to_gtsam_id[lm_id]


    def optimize(self, new_keyframes, new_imu_factors, new_landmarks, initial_velocities=None, initial_bias=None):

        new_graph = gtsam.NonlinearFactorGraph()
        new_estimates = gtsam.Values()

        # 判断是否初始化
        is_initialized = (initial_velocities is not None and initial_bias is not None)

        for i, kf in enumerate(new_keyframes):
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            
            if is_initialized:
                T_wc = gtsam.Pose3(kf.get_global_pose())
                T_cb = gtsam.Pose3(np.linalg.inv(self.T_bc))
                T_wb = T_wc.compose(T_cb)
                velocity = initial_velocities[i:i+3]

                if i == 0:
                    bias = initial_bias
                else:
                    bias = self.latest_bias
            else:
                # TODO:正常情况
                pass
                # T_wb = gtsam.Pose3(kf.get_global_pose())
                # velocity = gtsam.Vector3(0, 0, 0)
                # bias = self.latest_bias

            new_estimates.insert(X(kf_gtsam_id), T_wb)
            new_estimates.insert(V(kf_gtsam_id), velocity)
            new_estimates.insert(B(kf_gtsam_id), bias)

            # 为第一帧添加先验因子
            if kf_gtsam_id == 0:
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*3 + [1e-4]*3))
                prior_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4] * 3))
                prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2]*3 + [1e-3]*3))
                new_graph.add(gtsam.PriorFactorPose3(X(0), T_wb, prior_pose_noise))
                new_graph.add(gtsam.PriorFactorVector(V(0), velocity, prior_vel_noise))
                new_graph.add(gtsam.PriorFactorConstantBias(B(0), bias, prior_bias_noise))

        # 添加IMU预积分因子
        for factor_data in new_imu_factors:
            kf_start_id = next(kf.get_id() for kf in new_keyframes if kf.get_timestamp() == factor_data['start_kf_timestamp'])
            kf_end_id = next(kf.get_id() for kf in new_keyframes if kf.get_timestamp() == factor_data['end_kf_timestamp'])

            gtsam_id1 = self._get_kf_gtsam_id(kf_start_id)
            gtsam_id2 = self._get_kf_gtsam_id(kf_end_id)
            pim = factor_data['imu_preintegration']

            imu_factor = gtsam.CombinedImuFactor(
                X(gtsam_id1), V(gtsam_id1), X(gtsam_id2), V(gtsam_id2), 
                B(gtsam_id1), B(gtsam_id2), pim)
            new_graph.add(imu_factor)

        # 添加视觉因子
        visual_factor_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)
        for lm_id, lm_3d_pos in new_landmarks.items():
            if not new_estimates.exists(L(lm_id)):
                lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
                new_estimates.insert(L(lm_gtsam_id), lm_3d_pos)

        # 添加重投影因子
        for kf in new_keyframes:
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                T_bc_gtsam = gtsam.Pose3(self.T_bc)
                # 只处理本次优化中新添加的landmark
                if lm_id in new_landmarks:
                    lm_gtsam_id = self._get_lm_gtsam_id(lm_id)

                    visual_factor = gtsam.GenericProjectionFactorCal3_S2(
                        pt_2d, visual_factor_noise, X(kf_gtsam_id), L(lm_gtsam_id), 
                        self.K, body_P_sensor=T_bc_gtsam)
                    new_graph.add(visual_factor)

        # 执行iSAM2更新
        print(f"【Backend】: Updating iSAM2 with {new_graph.size()} new factors and {new_estimates.size()} new values...")
        self.isam2.update(new_graph, new_estimates)
        # 多迭代几次以保证收敛
        for _ in range(2): self.isam2.update()

        current_result = self.isam2.calculateEstimate()
        print("【Backend】: Optimization complete.")