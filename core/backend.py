import queue
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B, L
import re
from utils.debug import Debugger

class Backend:
    def __init__(self, global_central_map, config):
        self.global_central_map = global_central_map
        self.config = config
        
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
        self.T_cb = np.linalg.inv(self.T_bc)
        self.body_T_cam = gtsam.Pose3(self.T_bc)
        self.cam_T_body = gtsam.Pose3(self.T_cb)

        # 存储最新的优化后的偏置，用于IMU预积分
        self.latest_bias = gtsam.imuBias.ConstantBias()

        # 各种噪声参数
        self.visual_factor_noise = gtsam.noiseModel.Isotropic.Sigma(2, 3.0)
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
        self.prior_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*3))
        self.prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2]*6))

        # 鲁棒核函数
        # huber_k = 1.345
        # huber_kernel = gtsam.noiseModel.mEstimator.Huber(huber_k)
        # self.combined_noise_model = gtsam.noiseModel.Robust.Create(huber_kernel, self.visual_factor_noise)

        # 记录优化误差，debug用
        self.logger = Debugger(file_prefix="backend", column_names=["error"])

    def update_estimator_map(self, result, keyframe_window, active_landmarks, kf_id_to_gtsam_idx):
        print("【Backend】: Syncing optimized results back to Estimator...")
        # 更新关键帧位姿
        for kf in keyframe_window:
           # 获取待更新关键帧的gtsam_id
           kf_id = kf.get_id()
           gtsam_idx = kf_id_to_gtsam_idx.get(kf_id)

           if gtsam_idx is not None and result.exists(X(gtsam_idx)):
            # 获取优化后的位姿并转换回相机位姿
            pose_w_b = result.atPose3(X(gtsam_idx))
            pose_w_c = pose_w_b.compose(self.body_T_cam)

            print(f"pose_w_c: {pose_w_c.matrix()}")
            kf.set_global_pose(pose_w_c.matrix())
            
            # 更新关键帧速度、IMU偏置
            print(f"velocity: {result.atVector(V(gtsam_idx))}")
            kf.set_velocity(result.atVector(V(gtsam_idx)))

            print(f"bias: {result.atConstantBias(B(gtsam_idx))}")
            kf.set_bias(result.atConstantBias(B(gtsam_idx)))  

        for lm_id, landmark_obj in active_landmarks.items():
            if result.exists(L(lm_id)):
                optimized_position = result.atPoint3(L(lm_id))
                landmark_obj.set_triangulated(optimized_position)
                # print(f"lm_id {lm_id} optimized_position {optimized_position}")

        print("【Backend】: Syncing optimized results back to Estimator complete.")

    def optimize(self, keyframe_window, imu_factors, active_landmarks, initial_state_guess):
        print(f"【Backend】: Starting optimization for a window of {len(keyframe_window)} KFs and {len(active_landmarks)} LMs.")
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        kf_to_gtsam_idx = {kf.get_id(): i for i, kf in enumerate(keyframe_window)}

        for kf in keyframe_window:
            gtsam_idx = kf_to_gtsam_idx[kf.get_id()]

            pose_guess_mat, vel_guess, bias_guess = initial_state_guess[kf.get_id()]

            T_wc = gtsam.Pose3(pose_guess_mat)
            T_wb = T_wc.compose(self.cam_T_body)
            velocity = vel_guess
            bias = bias_guess

            values.insert(X(gtsam_idx), T_wb)
            values.insert(V(gtsam_idx), velocity)
            values.insert(B(gtsam_idx), bias)

        # 固定整个局部窗口的参考系
        first_kf_idx = 0
        graph.add(gtsam.PriorFactorPose3(X(first_kf_idx), values.atPose3(X(first_kf_idx)), self.prior_pose_noise))
        graph.add(gtsam.PriorFactorVector(V(first_kf_idx), values.atVector(V(first_kf_idx)), self.prior_vel_noise))
        graph.add(gtsam.PriorFactorConstantBias(B(first_kf_idx), values.atConstantBias(B(first_kf_idx)), self.prior_bias_noise))

        # 添加IMU因子
        for factor_data in imu_factors:
            start_kf_id = factor_data['start_kf_id']
            end_kf_id = factor_data['end_kf_id']

            if start_kf_id in kf_to_gtsam_idx and end_kf_id in kf_to_gtsam_idx:
                gtsam_id1 = kf_to_gtsam_idx[start_kf_id]
                gtsam_id2 = kf_to_gtsam_idx[end_kf_id]
                pim = factor_data['imu_preintegration']
                graph.add(gtsam.CombinedImuFactor(X(gtsam_id1), V(gtsam_id1), X(gtsam_id2), V(gtsam_id2), B(gtsam_id1), B(gtsam_id2), pim))
        
        # 添加路标点重投影视觉因子
        for lm_id, landmark_obj in active_landmarks.items():
            lm_gtsam_id = lm_id

            valid_observations = []
            for obs_kf_id, obs_pt_2d in landmark_obj.get_all_observations().items():
                if obs_kf_id in kf_to_gtsam_idx:
                    valid_observations.append((obs_kf_id, obs_pt_2d))

            # print(f"valid_observations: {len(valid_observations)}")
            if len(valid_observations) < 2:
                continue

            values.insert(L(lm_gtsam_id), landmark_obj.get_position())

            for obs_kf_id, obs_pt_2d in valid_observations:
                kf_gtsam_id = kf_to_gtsam_idx[obs_kf_id]
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    obs_pt_2d, self.visual_factor_noise, X(kf_gtsam_id), L(lm_gtsam_id), 
                    self.K, body_P_sensor=self.body_T_cam)
                graph.add(factor)

        # 执行优化
        print(f"【Backend】: Optimizing graph with {graph.size()} factors and {values.size()} values...")
        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosityLM("ERROR")
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
            result = optimizer.optimize()
            print("【Backend】: Optimization complete.")

            # 记录优化误差
            final_error = optimizer.error()
            print(f"【Backend】: Final error: {final_error:.4f}")

        except Exception as e:
            print(f"【Backend】: Optimization failed! Error: {e}")
            return False


        # 获取优化结果,这里一样用函数更新estimator_map的方法来更新
        self.update_estimator_map(result, keyframe_window, active_landmarks, kf_to_gtsam_idx)

        return True