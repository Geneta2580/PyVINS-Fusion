import queue
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B

class Backend:
    def __init__(self, global_central_map, config, imu_processor):
        self.global_central_map = global_central_map
        self.config = config
        self.imu_processor = imu_processor

        # 传递submap和imu预积分因子的队列
        self.measurement_queue = queue.Queue()

        # FIX 1: 使用 iSAM2 作为优化器
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1) 
        parameters.relinearizeSkip = 1
        self.isam2 = gtsam.ISAM2(parameters)
        
        self.timestamps_to_gtsam_id = {}
        self.next_id = 0
        self.map_lock = threading.Lock()
        
        # 获取相机IMU外参
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)

        # 存储最新的优化后的偏置，用于IMU预积分
        self.latest_bias = gtsam.imuBias.ConstantBias()
        self.latest_bias_timestamp = None

    def stop(self):
        self.stop_event.set()

    def add_new_measurements(self, new_keyframes, imu_factors,  initial_velocities=None, initial_bias=None):
        self.measurement_queue.put({
            'keyframes': new_keyframes, 
            'imu_factors': imu_factors, 
            'initial_velocities': initial_velocities, 
            'initial_bias': initial_bias
        })

    def run(self):
        print("【Backend】: GTSAM-iSAM2 thread started. Waiting for triggers...")
        while not self.stop_event.is_set():
            try:
                # 阻塞式操作
                task = self.measurement_queue.get(block=True, timeout=1.0)

                # 解包数据
                new_keyframes = task['keyframes']
                imu_factors = task['imu_factors']
                initial_velocities = task['initial_velocities']
                initial_bias = task['initial_bias']

                print(f"【Backend】: Received task with {len(new_keyframes)} KFs and {len(imu_factors)} IMU factors.")
                self.optimize_new_submap(new_keyframes, imu_factors, initial_velocities, initial_bias)

            except queue.Empty:
                continue # 队列为空，继续等待
            except Exception as e:
                print(f"!!!!!! EXCEPTION IN BACKEND THREAD !!!!!!")
                import traceback
                traceback.print_exc()
        print("【Backend】: GTSAM-iSAM2 thread stopped.")

    def get_or_create_id(self, timestamp):
        if timestamp not in self.timestamps_to_gtsam_id:
            self.timestamps_to_gtsam_id[timestamp] = self.next_id
            self.next_id += 1
        return self.timestamps_to_gtsam_id[timestamp]

    def optimize_new_submap(self, new_keyframes, new_imu_factors, initial_velocities, initial_bias):
        # 为新节点和因子创建临时的图和初始值
        new_graph = gtsam.NonlinearFactorGraph()
        new_estimates = gtsam.Values()
        
        # 判断是否为一个个submap
        is_initial_batch = (initial_velocities is not None and initial_bias is not None)

        # 获取已优化的最新状态
        last_isam_state = self.isam2.calculateEstimate()

        # 存储本次优化和上次优化过程中的估计值
        working_estimates = gtsam.Values(last_isam_state)

        for i, kf in enumerate(new_keyframes):
            # 为每一个KF映射id
            ts = kf.get_timestamp()
            T_w_ci = kf.get_global_pose() # 这里获取的是T_w_ci
            T_w_bi = T_w_ci @ np.linalg.inv(self.T_bc)

            # 创建KF_id，也是gtsam的graph_id
            kf_id = self.get_or_create_id(ts)

            if kf_id == 0:
                # 获取初始位姿
                print(f"【Backend】: Initial pose: {T_w_bi}")

                initial_pose = gtsam.Pose3(T_w_bi)

                v0 = initial_velocities[0:3] if is_initial_batch else np.zeros(3)
                b0 = initial_bias if is_initial_batch else gtsam.imuBias.ConstantBias()

                # 为第一个状态量添加入初始值
                new_estimates.insert(X(0), initial_pose)
                new_estimates.insert(V(0), v0)
                new_estimates.insert(B(0), b0)

                working_estimates.insert(X(0), initial_pose)
                working_estimates.insert(V(0), v0)
                working_estimates.insert(B(0), b0)

                # 添加极小的协方差来固定第一帧的Pose
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
                prior_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 3))
                prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4] * 6))

                # 添加第一帧的先验因子
                new_graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, prior_pose_noise))
                new_graph.add(gtsam.PriorFactorVector(V(0), v0, prior_vel_noise))
                new_graph.add(gtsam.PriorFactorConstantBias(B(0), b0, prior_bias_noise))
                continue

            # 添加IMU因子边
            imu_factor_info = next((f for f in new_imu_factors if f['end_kf_timestamp'] == ts), None)

            # 初始化预测值，默认使用零速度和零偏置
            predicted_vel = np.zeros(3)
            predicted_bias = gtsam.imuBias.ConstantBias()
            
            # 若有imu因子，则使用IMU预测
            if imu_factor_info:
                # 映射上一个KF的id
                prev_ts = imu_factor_info['start_kf_timestamp']                
                start_id = self.get_or_create_id(prev_ts)

                # 为新状态量提供初始估计值(节点)，使用上一帧状态来预测新状态，计算预测值
                prev_pose = working_estimates.atPose3(X(start_id))
                prev_vel = working_estimates.atVector(V(start_id))
                prev_bias = working_estimates.atConstantBias(B(start_id))

                imu_preintegration = imu_factor_info['imu_preintegration']

                # --- 【DEBUG代码】 ---
                print(f"【DEBUG-IMU Factor】")
                print(f"  Start TS: {prev_ts:.6f}, End TS: {ts:.6f}")
                print(f"  前端 IMU 积分姿态 (R): \n{imu_preintegration.deltaRij().matrix()}")
                print(f"  前端 IMU 积分位置 (P): {imu_preintegration.deltaPij()}")
                print(f"  前端 IMU 积分速度 (V): {imu_preintegration.deltaVij()}")

                predicted_state = imu_preintegration.predict(gtsam.NavState(prev_pose, prev_vel), prev_bias)
                print("running1")
                predicted_vel = predicted_state.velocity()
                print("running2")
                predicted_bias = prev_bias
                print("running3")


                # 计算IMU预测的位移大小
                imu_displacement = np.linalg.norm(predicted_state.pose().translation() - prev_pose.translation())
                # 计算视觉测量的位移大小
                visual_displacement = np.linalg.norm(gtsam.Pose3(T_w_bi).translation() - prev_pose.translation())

                print(f"--- MOTION CHECK ---")
                print(f"  Displacement from IMU:    {imu_displacement:.4f} meters")
                print(f"  Displacement from Visual: {visual_displacement:.4f} meters")
                print(f"  Scale Ratio (Visual/IMU): {visual_displacement / imu_displacement if imu_displacement > 1e-6 else 0:.4f}")

                # 2. 对比旋转
                # 视觉测量的相对旋转
                T_visual_relative = prev_pose.inverse().compose(gtsam.Pose3(T_w_bi))
                R_visual = T_visual_relative.rotation()
                
                # IMU测量的相对旋转
                R_imu = imu_preintegration.deltaRij()
                
                # 计算两个旋转之间的差异
                R_diff = R_visual.inverse().compose(R_imu)

                print("R_diff: \n", R_diff.matrix())

                print(f"  上一状态 (X{start_id}) 优化结果: Pose={prev_pose.translation()}, Vel={prev_vel}")
                print(f"  上一状态 (X{start_id}) 优化结果: Bias={prev_bias}")
                print(f"  当前状态 (X{kf_id}) IMU 预测 Pose: {predicted_state.pose().translation()}")
                print(f"  当前状态 (X{kf_id}) IMU 预测 Velocity: {predicted_vel}")
                # --- 【DEBUG代码结束】 ---                

                # 创建IMU因子(边)
                imu_factor = gtsam.CombinedImuFactor(
                    X(start_id), V(start_id), X(kf_id), V(kf_id), B(start_id), B(kf_id),
                    imu_preintegration
                )
                
                new_graph.add(imu_factor)

            if is_initial_batch:
                current_vel = initial_velocities[i*3:i*3+3]
                current_bias = initial_bias
            else:
                current_vel = predicted_vel
                current_bias = predicted_bias

            # 使用视觉位姿作为位置初值，IMU速度初值，IMU偏置不变，将预测值放入估计值中
            new_estimates.insert(X(kf_id), gtsam.Pose3(T_w_bi))
            new_estimates.insert(V(kf_id), current_vel)
            new_estimates.insert(B(kf_id), current_bias) # 偏置通常假设为缓慢变化
            
            working_estimates.insert(X(kf_id), gtsam.Pose3(T_w_bi))
            working_estimates.insert(V(kf_id), current_vel)
            working_estimates.insert(B(kf_id), current_bias)

            # # 视觉约束因子
            # if i > 0:
            #     prev_kf_in_batch = new_keyframes[i-1]
            #     prev_kf_id = self.get_or_create_id(prev_kf_in_batch.get_timestamp())
                
            #     # T_prev_curr = T_world_prev^-1 * T_world_curr (从局部坐标系计算相对位姿)
            #     T_prev_curr_gtsam = gtsam.Pose3(np.linalg.inv(prev_kf_in_batch.get_local_pose()) @ kf.get_local_pose())
                
            #     # 视觉测量噪声模型
            #     visual_noise = gtsam.noiseModel.Diagonal.Sigmas(
            #         np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2]) # 旋转xyz标准差, 平移xyz标准差 (米)
            #     )
            #     new_graph.add(gtsam.BetweenFactorPose3(
            #         X(prev_kf_id), X(kf_id), T_prev_curr_gtsam, visual_noise
            #     ))

        if not new_graph.empty():
            # 1. 计算更新前的误差
            # 首先获取 iSAM2 中已有的优化结果
            last_result = self.isam2.calculateEstimate()
            estimates_for_error_check = gtsam.Values(last_result) # 创建一个临时的、完备的 Values 对象用于计算误差
            estimates_for_error_check.insert(new_estimates) # 将新的估计值插入到临时对象中
                 
            error_before = new_graph.error(estimates_for_error_check)
            print(f"【Backend-DEBUG】: Error BEFORE iSAM2 update (for new factors): {error_before:.4f}")

            # 2. 执行 iSAM2 更新
            print(f"【Backend】: Updating iSAM2 with {len(new_imu_factors)} new IMU factors...")
            self.isam2.update(new_graph, new_estimates)
            current_result = self.isam2.calculateEstimate()
            
            # 3. 计算更新后的误差
            error_after = new_graph.error(current_result) # 使用刚刚得到的最新优化结果来计算新图的误差
            print(f"【Backend-DEBUG】: Error AFTER iSAM2 update (for new factors):  {error_after:.4f}")
            print(f"【Backend-DEBUG】: Error reduction for this step: {error_before - error_after:.4f}")
            # --- 核心逻辑结束 ---

            with self.map_lock:
                self.update_global_map(current_result)
                
                # 更新最新的偏置估计，用于下一次IMU预积分
                if len(new_keyframes) > 0:
                    last_kf_ts = new_keyframes[-1].get_timestamp()
                    last_kf_id = self.get_or_create_id(last_kf_ts)
                    if current_result.exists(B(last_kf_id)):
                        self.latest_bias = current_result.atConstantBias(B(last_kf_id))
                        self.latest_bias_timestamp = last_kf_ts
                        print(f"【Backend】: Updated latest bias at timestamp {last_kf_ts}")
                        
            print("【Backend】: iSAM2 update finished.")

    def update_global_map(self, optimized_results):
        print("【Backend】: Updating global map with iSAM2 results...")
        updated_count = 0
        for ts, kf_id in self.timestamps_to_gtsam_id.items():
            if optimized_results.exists(X(kf_id)):
                # 从优化结果中获取最新的位姿
                optimized_pose_gtsam = optimized_results.atPose3(X(kf_id))
                optimized_pose_np = optimized_pose_gtsam.matrix() # 这里获取的是T_w_bi
                optimized_pose_np = optimized_pose_np @ self.T_bc # 切换回T_w_ci

                # 直接调用GlobalMap的高效更新接口
                self.global_central_map.update_keyframe_pose(ts, optimized_pose_np)
                updated_count += 1
        
        print(f"【Backend】: Global map update complete. Updated {updated_count} keyframes.")
    
    def get_latest_bias(self):
        with self.map_lock:
            return self.latest_bias