import queue
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B, L
from gtsam_unstable import IncrementalFixedLagSmoother, FixedLagSmootherKeyTimestampMap

import re
from utils.debug import Debugger
import time

class Backend:
    def __init__(self, global_central_map, config, imu_processor):
        self.global_central_map = global_central_map
        self.config = config

        # 使用 iSAM2 作为优化器
        self.lag_window_size = config.get('lag_window_size', 10) # 优化器的滑窗
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01) 
        parameters.relinearizeSkip = 1
        self.smoother = IncrementalFixedLagSmoother(self.lag_window_size, parameters) # 自动边缘化
        
        # 鲁棒因子
        self.visual_noise = gtsam.noiseModel.Isotropic.Sigma(2, 3.0)
        self.visual_robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.visual_noise)

        # 状态与id管理
        self.kf_id_to_gtsam_id = {}
        self.landmark_id_to_gtsam_id = {}
        self.next_gtsam_kf_id = 0
        self.factor_indices_to_remove = gtsam.KeyVector()

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

        # 定义要记录的列
        log_columns = [
            "gtsam_id", "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "bias_acc_x", "bias_acc_y", "bias_acc_z",
            "bias_gyro_x", "bias_gyro_y", "bias_gyro_z",
            "new_factors_error"
        ]
        # 初始化Debugger
        self.logger = Debugger(file_prefix="backend_state", column_names=log_columns)

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

        result = self.smoother.calculateEstimate()

        try:
            pose = result.atPose3(X(latest_gtsam_id))
            velocity = result.atVector(V(latest_gtsam_id))
            bias = result.atConstantBias(B(latest_gtsam_id))
            # print(f"【Backend】: Latest optimized state: pose: {pose.matrix()}, velocity: {velocity}, bias: {bias}")
            return pose, velocity, bias
        except Exception as e:
            print(f"[Error][Backend] Failed to retrieve latest state for gtsam_id {latest_gtsam_id}: {e}")
            return None, None, None

    def update_estimator_map(self, keyframe_window, landmarks):
        print("【Backend】: Syncing optimized results back to Estimator...")
        optimized_results = self.smoother.calculateEstimate()

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
        for lm_id, landmark_obj in landmarks.items():
            gtsam_id = self._get_lm_gtsam_id(lm_id)
            if gtsam_id is not None and optimized_results.exists(L(gtsam_id)):
                # 1. 从优化结果中获取最新的3D坐标
                optimized_position = optimized_results.atPoint3(L(gtsam_id))
                # 2. 调用对象的方法来更新其内部状态
                landmark_obj.set_triangulated(optimized_position)
                # print(f"【Backend】: Updated landmark {lm_id} to {optimized_position}")

    def remove_stale_landmarks(self, unhealty_lm_ids, unhealty_lm_ids_depth, oldest_kf_id_in_window):
        # 2. 修改：这个函数现在只负责“登记”要删除的路标点ID，不计算索引
        print(f"【Backend】: 接收到移除 {len(unhealty_lm_ids)} 个陈旧路标点的指令。")
        if not unhealty_lm_ids:
            return

        graph = self.smoother.getFactors()
        factor_indices_to_remove = gtsam.KeyVector()
        unhealty_lm_keys = {L(self._get_lm_gtsam_id(lm_id)) for lm_id in unhealty_lm_ids}

        factor_indices_to_remove_depth = gtsam.KeyVector()
        unhealty_lm_keys_depth = {L(self._get_lm_gtsam_id(lm_id)) for lm_id in unhealty_lm_ids_depth}

        # oldest_kf_id_in_window += 1 
        oldest_gtsam_key = None
        if oldest_kf_id_in_window is not None and oldest_kf_id_in_window in self.kf_id_to_gtsam_id:
            oldest_gtsam_key = X(self._get_kf_gtsam_id(oldest_kf_id_in_window))
            print(f"【Backend】: 最旧的关键帧的gtsam_id: {oldest_gtsam_key}")


        for i in range(graph.size()):
            factor = graph.at(i)
            if factor is not None:
                for key in factor.keys():
                    # print(f"key: {key}")
                    # if key == oldest_gtsam_key:
                    #     factor_type = factor.__class__.__name__
                    #     key_str = ", ".join([gtsam.DefaultKeyFormatter(key) for key in factor.keys()])
                    #     print(f"  [跳过删除] Index: {i}, 类型: {factor_type}, 连接: [{key_str}] (这是最旧的关键帧)")
                    #     break # 绝不删除与最旧的关键帧相连的因子

                    if key in unhealty_lm_keys:
                        factor_type = factor.__class__.__name__
                        key_str = ", ".join([gtsam.DefaultKeyFormatter(key) for key in factor.keys()])
                        print(f"  [标记删除] Index: {i}, 类型: {factor_type}, 连接: [{key_str}]")
                        factor_indices_to_remove.append(i)

                        if key in unhealty_lm_keys_depth:
                            print(f"检测到深度为负的因子，尝试删除")
                            # print(f"key: {key}")
                            print(f"  [标记删除深度] Index: {i}, 类型: {factor_type}, 连接: [{key_str}]")
                            if factor_type != 'GenericProjectionFactorCal3_S2':
                                print(f"  [跳过删除] Index: {i}, 类型: {factor_type} (这是边缘化锚点)")
                                continue # 绝不删除边缘化因子
                            
                            factor_indices_to_remove_depth.append(i)
                            print(f"  [确认删除深度] Index: {i}, 类型: {factor_type}, 连接: [{key_str}]")
                        break
        
        self.factor_indices_to_remove = factor_indices_to_remove_depth

        if factor_indices_to_remove_depth:
            empty_graph = gtsam.NonlinearFactorGraph()
            empty_values = gtsam.Values()
            empty_stamps = FixedLagSmootherKeyTimestampMap()
            self.smoother.update(empty_graph, empty_values, empty_stamps, factor_indices_to_remove_depth)
            print(f"【Backend】: 成功移除 {len(factor_indices_to_remove_depth)} 个深度为负的路标点的因子")

        for lm_id in unhealty_lm_ids:
            if lm_id in self.landmark_id_to_gtsam_id:
                del self.landmark_id_to_gtsam_id[lm_id]

        print(f"【Backend】: 成功移除 {len(unhealty_lm_ids)} 个路标点的因子")
        

    def initialize_optimize(self, initial_keyframes, initial_imu_factors, initial_landmarks, initial_velocities, initial_bias):
        print("【Backend】: Initializing optimize...")

        graph = gtsam.NonlinearFactorGraph()
        estimates = gtsam.Values()
        
        initial_window_stamps = FixedLagSmootherKeyTimestampMap()

        for i, kf in enumerate(initial_keyframes):
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())

            # 从初始化结果中获取位姿、速度和偏置
            T_wc = gtsam.Pose3(kf.get_global_pose())
            T_wb = T_wc.compose(self.cam_T_body)
            # initial_velocities 是一个扁平化的数组，每3个元素是一个速度向量
            velocity = initial_velocities[i*3 : i*3+3]
            
            # 所有帧使用相同的初始偏置
            bias = initial_bias

            # 添加初始估计值
            estimates.insert(X(kf_gtsam_id), T_wb)
            estimates.insert(V(kf_gtsam_id), velocity)
            estimates.insert(B(kf_gtsam_id), bias)

            # 添加滑窗记录
            initial_window_stamps.insert((X(kf_gtsam_id), float(kf_gtsam_id)))
            initial_window_stamps.insert((V(kf_gtsam_id), float(kf_gtsam_id)))
            initial_window_stamps.insert((B(kf_gtsam_id), float(kf_gtsam_id)))

            # 为每一个landmark设置滑窗记录
            last_gtsam_id = self._get_kf_gtsam_id(initial_keyframes[-1].get_id())
            for lm_id in initial_landmarks.keys():
                lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
                initial_window_stamps.insert((L(lm_gtsam_id), float(last_gtsam_id))) # 设为最后一帧的ID

            # 为第一帧添加强先验
            if kf_gtsam_id == 0:
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*3 + [1e-2]*3))
                prior_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2] * 3))
                prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1]*3 + [1e-2]*3))
                graph.add(gtsam.PriorFactorPose3(X(0), T_wb, prior_pose_noise))
                graph.add(gtsam.PriorFactorVector(V(0), velocity, prior_vel_noise))
                graph.add(gtsam.PriorFactorConstantBias(B(0), bias, prior_bias_noise))
            elif kf_gtsam_id == len(initial_keyframes) - 1:
                # 为最后一帧添加较弱的位置先验以稳定尺度
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*3 + [0.5]*3))
                graph.add(gtsam.PriorFactorPose3(X(kf_gtsam_id), T_wb, prior_pose_noise))

        # 添加所有初始IMU因子
        for factor_data in initial_imu_factors:
            start_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['start_kf_timestamp'])
            end_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['end_kf_timestamp'])
            gtsam_id1 = self._get_kf_gtsam_id(start_kf.get_id())
            gtsam_id2 = self._get_kf_gtsam_id(end_kf.get_id())
            pim = factor_data['imu_preintegration']
            graph.add(gtsam.CombinedImuFactor(X(gtsam_id1), V(gtsam_id1), X(gtsam_id2), V(gtsam_id2), B(gtsam_id1), B(gtsam_id2), pim))

        # 添加所有初始路标点变量和视觉因子
        for lm_id, lm_3d_pos in initial_landmarks.items():
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            estimates.insert(L(lm_gtsam_id), lm_3d_pos)

        for kf in initial_keyframes:
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                # 只处理本次优化中新添加的landmark
                if lm_id in initial_landmarks:
                    lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
                    factor = gtsam.GenericProjectionFactorCal3_S2(pt_2d, self.visual_robust_noise, X(kf_gtsam_id), L(lm_gtsam_id), self.K, body_P_sensor=self.body_T_cam)
                    graph.add(factor)

        # 执行iSAM2的第一次更新（批量模式）
        print(f"【Backend】: Initializing iSAM2 with {graph.size()} new factors and {estimates.size()} new values...")
        
        try:
            start_time = time.time()
            self.smoother.update(graph, estimates, initial_window_stamps)
            end_time = time.time()
            print(f"【Backend Timer】: Initial optimization took { (end_time - start_time) * 1000:.3f} ms.")
        except RuntimeError as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!! INITIALIZATION FAILED !!!!!!!!!!!!!!")
            print(f"ERROR: {e}")
            return # 失败时必须返回

        # 更新最新bias
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        print(f"【Backend】: Latest optimized state: pose: {latest_pose.matrix()}, velocity: {latest_vel}, bias: {latest_bias}")

        latest_gtsam_id = self.next_gtsam_kf_id - 1
        print(f"【Backend】: Latest gtsam_id: {latest_gtsam_id}")
        if latest_bias is not None:
            self.latest_bias = latest_bias
        print("【Backend】: Initial graph optimization complete.")

        # 记录优化状态
        new_factors_error = self._log_optimization_error(graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)


    def optimize_incremental(self, last_keyframe, new_keyframe, new_imu_factors, 
                            new_landmarks, new_visual_factors, initial_state_guess, is_stationary, oldest_kf_id_in_window):

        new_graph = gtsam.NonlinearFactorGraph()
        new_estimates = gtsam.Values()
        new_window_stamps = FixedLagSmootherKeyTimestampMap()

        # 添加新关键帧的状态变量，使用IMU预测值作为初始估计
        kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())
        T_wb_guess, vel_guess, bias_guess = initial_state_guess

        new_estimates.insert(X(kf_gtsam_id), T_wb_guess)
        new_estimates.insert(V(kf_gtsam_id), vel_guess)
        new_estimates.insert(B(kf_gtsam_id), bias_guess)

        # 添加滑窗记录
        new_window_stamps.insert((X(kf_gtsam_id), float(kf_gtsam_id)))
        new_window_stamps.insert((V(kf_gtsam_id), float(kf_gtsam_id)))
        new_window_stamps.insert((B(kf_gtsam_id), float(kf_gtsam_id)))

        # 添加IMU因子
        last_kf_gtsam_id = self._get_kf_gtsam_id(last_keyframe.get_id())
        pim = new_imu_factors['imu_preintegration']
        imu_factor = gtsam.CombinedImuFactor(
            X(last_kf_gtsam_id), V(last_kf_gtsam_id), X(kf_gtsam_id), V(kf_gtsam_id),
            B(last_kf_gtsam_id), B(kf_gtsam_id), pim)
        new_graph.add(imu_factor)

        # 添加新路标点顶点，注意这里添加的顶点只在new_estimates中还没有进入isam2的图
        for lm_id, lm_3d_pos in new_landmarks.items():
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            # ---!!!--- 在此处添加您要的日志 ---!!!---
            # 打印即将送入优化器的路标点的值
            # print(f"🕵️‍ 【Backend】: 优化器即将处理新路标点 L{lm_id}，其三角化初始值为: {lm_3d_pos}")
            
            # 增加一个NaN/Inf的显式检查，这对于调试崩溃至关重要
            if np.isnan(lm_3d_pos).any() or np.isinf(lm_3d_pos).any():
                print(f"🔥 【Backend】[致命警告]: 路标点 L{lm_id} 的初始值无效 (NaN/Inf)！优化即将因此崩溃！")
            # ---!!!--- 日志添加结束 ---!!!---

            # 检查：1) 不在旧图中，2) 还没被添加过 确保顶点只被添加一次
            if not self.smoother.calculateEstimate().exists(L(lm_gtsam_id)):
                new_estimates.insert(L(lm_gtsam_id), lm_3d_pos)
        
        # 添加重投影因子，前面已经添加了新路标点顶点，所以这里只需要添加历史点和新特征点的观测帧重投影因子
        current_isam_values = self.smoother.calculateEstimate()
        for kf_id, lm_id, pt_2d in new_visual_factors:
            kf_gtsam_id = self._get_kf_gtsam_id(kf_id)
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)

            kf_exists = current_isam_values.exists(X(kf_gtsam_id)) or new_estimates.exists(X(kf_gtsam_id))
            lm_exists = current_isam_values.exists(L(lm_gtsam_id)) or new_estimates.exists(L(lm_gtsam_id))

            if kf_exists and lm_exists:
                factor = gtsam.GenericProjectionFactorCal3_S2(pt_2d, self.visual_robust_noise, X(kf_gtsam_id), L(lm_gtsam_id), self.K, body_P_sensor=self.body_T_cam)
                new_graph.add(factor)
                new_window_stamps.insert((L(lm_gtsam_id), float(kf_gtsam_id))) # 这里也需要更新历史路标点的滑窗记录


        # ======================= ZERO-VELOCITY UPDATE (ZUPT) - SOFT CONSTRAINT =======================
        if is_stationary:
            kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())
            zero_velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)
            zero_velocity_prior = gtsam.PriorFactorVector(V(kf_gtsam_id), np.zeros(3), zero_velocity_noise)
            new_graph.add(zero_velocity_prior)
            print("【Backend】: Added Zero-Velocity-Update (ZUPT) factor.")
            # # 使用BetweenFactor在连续两帧的速度之间添加一个软约束，使它们趋于一致（即速度变化为零）
            # last_kf_gtsam_id = self._get_kf_gtsam_id(last_keyframe.get_id())
            # new_kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())

            # # 噪声模型相对宽松，允许一定的抖动
            # zero_velocity_diff_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1) 
            
            # # 约束 V(new) - V(last) = 0
            # zupt_factor = gtsam.BetweenFactorVector(V(last_kf_gtsam_id), 
            #                                         V(new_kf_gtsam_id), 
            #                                         np.zeros(3), 
            #                                         zero_velocity_diff_noise)
            # new_graph.add(zupt_factor)
            # print("【Backend】: Added soft Zero-Velocity-Update (ZUPT) factor between frames.")
        # ============================================================================================

        # 执行iSAM2增量更新
        print(f"【Backend】: Updating iSAM2 ({new_graph.size()} new factors, {new_estimates.size()} new variables)...")
        
        try:
            start_time = time.time()
            self.smoother.update(new_graph, new_estimates, new_window_stamps)
            end_time = time.time()
            print(f"【Backend Timer】: Incremental optimization took { (end_time - start_time) * 1000:.3f} ms.")

        except RuntimeError as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!! OPTIMIZATION FAILED !!!!!!!!!!!!!!")
            print(f"ERROR: {e}")
            return

        # 更新最新bias
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        print(f"【Backend】: Latest optimized state: pose: {latest_pose.matrix()}, velocity: {latest_vel}, bias: {latest_bias}")
        latest_gtsam_id = self.next_gtsam_kf_id - 1
        if latest_bias is not None:
            self.latest_bias = latest_bias

        # 记录优化误差
        new_factors_error = self._log_optimization_error(new_graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        print("【Backend】: Incremental optimization complete.")


    def _log_optimization_error(self, new_factors_graph):
        try:
            optimized_result = self.smoother.calculateEstimate()
            new_factors_error = new_factors_graph.error(optimized_result)

            current_full_graph = self.smoother.getFactors()

            print(f"【Backend】优化误差统计: "
                  f"本轮新增因子误差 = {new_factors_error:.4f}")

            # ======================= DETAILED FACTOR ERROR LOGGING =======================
            debug_start_frame = 600 # 设为0以立即开始打印
            latest_gtsam_id = self.next_gtsam_kf_id - 1
            if latest_gtsam_id >= debug_start_frame:
                print("\n" + "="*40 + f" DETAILED ERROR ANALYSIS (Frame {latest_gtsam_id}) " + "="*40)
                
                # 遍历图中的所有因子
                for i in range(current_full_graph.size()):
                    factor = current_full_graph.at(i)
                    if factor is None: # 检查因子是否有效
                        continue
                        
                    try:
                        # 计算这个特定因子的误差
                        error = factor.error(optimized_result)
                        
                        # 打印误差大于阈值的因子，以避免日志刷屏
                        if error > 100.0: 
                            # 打印因子的Python类名
                            factor_type = factor.__class__.__name__
                            print(f"  - Factor {i}: Error = {error:.4f}, Type = {factor_type}")
                            
                            # 尝试打印与该因子相关的Key
                            keys = factor.keys()
                            key_str = ", ".join([gtsam.DefaultKeyFormatter(key) for key in keys])
                            print(f"    Keys: [{key_str}]")
                            
                    except Exception as e_factor:
                        # 捕获计算单个因子误差时可能发生的错误
                        print(f"  - Factor {i}: 无法计算误差或获取Keys. Error: {e_factor}")

                print("="*100 + "\n")
            # ===========================================================================
            
            return new_factors_error
            
        except Exception as e:
            print(f"[Error][Backend] 计算优化误差时出错: {e}")
            return -1.0, -1.0
        
    def _log_state_and_errors(self, latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error):
        position = latest_pose.translation()
        acc_bias = latest_bias.accelerometer()
        gyro_bias = latest_bias.gyroscope()

        state_data = {
            "gtsam_id": latest_gtsam_id,
            "pos_x": position[0], "pos_y": position[1], "pos_z": position[2],
            "vel_x": latest_vel[0], "vel_y": latest_vel[1], "vel_z": latest_vel[2],
            "bias_acc_x": acc_bias[0], "bias_acc_y": acc_bias[1], "bias_acc_z": acc_bias[2],
            "bias_gyro_x": gyro_bias[0], "bias_gyro_y": gyro_bias[1], "bias_gyro_z": gyro_bias[2],
            "new_factors_error": new_factors_error
        }
        self.logger.log_state(state_data)
