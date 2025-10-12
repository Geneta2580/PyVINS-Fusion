import gtsam
import numpy as np
import torch
import threading
import queue

from .backend import Backend
from .keyframe import KeyFrame
from .submap import Submap
from .global_map import GlobalMap
from utils.geometry import align_submaps, pose_matrix_to_tum_format
from .vio_initializer import VIOInitializer

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map

class IVGGTSystem:
    """
    The central coordinator for the SLAM system. Runs as a consumer thread.
    """
    def __init__(self, config, input_queue, viewer_queue, global_central_map, imu_processor, backend):
        self.config = config
        self.input_queue = input_queue
        self.viewer_queue = viewer_queue

        # 从配置文件中读取置信度阈值，如果未定义则使用默认值0.9
        self.confidence_threshold = self.config.get('confidence_threshold', 0.9)
        # print(f"Point cloud confidence threshold set to: {self.confidence_threshold}")
        
        # Viewer and Map are managed here, but viewer is passed in from main for thread control
        self.viewer = None
        self.backend = backend
        self.global_map = global_central_map
        
        self.submap_counter = 0
        self.submaps = {}

        # 初始化相关设置
        self.is_initialized = False
        self.init_kf_buffer = []
        self.init_imu_factors_buffer = []
        self.init_window_size = self.config.get('init_window_size', 10)
        self.imu_processor = imu_processor

        # Threading control
        self.is_running = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def set_viewer(self, viewer):
        self.viewer = viewer

    def start(self):
        """Starts the system thread."""
        self.is_running = True
        self.thread.start()

    def shutdown(self):
        """Stops the system thread and saves the trajectory."""
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join()
        
        # --- 核心改动：从最终的 global_map 生成轨迹 ---
        print("Generating final trajectory from optimized keyframes...")
        final_trajectory = []
        # 从全局地图中获取所有经过优化的keyframe
        all_keyframes = sorted(self.global_map.get_all_keyframes(), key=lambda kf: kf.get_timestamp())

        for kf in all_keyframes:
            timestamp = kf.get_timestamp()
            pose = kf.get_global_pose() # poses is an array of (4, 4) matrices

            tx, ty, tz, qx, qy, qz, qw = pose_matrix_to_tum_format(pose)
            ts_sec = timestamp
            tum_line = f"{ts_sec:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
            final_trajectory.append(tum_line)

        if final_trajectory:
            output_path = self.config.get('trajectory_output_path', 'estimated_trajectory.txt')
            with open(output_path, "w") as f:
                for line in final_trajectory:
                    f.write(line + "\n")
            print(f"Optimized trajectory saved to {output_path}")

        print("System shut down.")

    def run(self):
        """Main loop for the system thread, consuming data from the input queue."""
        print("System thread started.")
        while self.is_running:
            try:
                package = self.input_queue.get(timeout=1.0)

                if package is None:
                    print("System received shutdown signal from frontend.")
                    break 

                if not self.is_initialized:
                    self.visual_inertial_initialization(package)
                else:
                    # pass
                    self.process_package_data(package)

            except queue.Empty:
                continue
        
        self.is_running = False
        print("System thread has finished.")

    def visual_inertial_initialization(self, package):
        visual_predictions = package['visual_predictions']
        imu_measurements = package['imu_measurements']

        if not visual_predictions or not imu_measurements:
            print("No visual predictions or imu measurements")
            return

        # 对IMU量测值进行预积分
        imu_factors = self.process_imu_data(imu_measurements, use_latest_bias=False)

        prev_submap = self.submaps.get(self.submap_counter - 1)
        new_keyframes, scale = self.create_and_align_keyframes(visual_predictions, prev_submap)
        
        # print(f"【System Init】: new_keyframes[0].get_timestamp(): {new_keyframes[0].get_timestamp()}")
        # print(f"【System Init】: new_keyframes[0].get_global_poses(): {new_keyframes[0].get_global_poses()}")

        if not new_keyframes:
            return

        temp_submap = Submap(self.submap_counter)
        temp_submap.set_scale(scale)

        for kf in new_keyframes:
            temp_submap.add_keyframe(kf)
            self.global_map.add_keyframe(kf) # 添加到global_map

        self.submaps[self.submap_counter] = temp_submap
        self.submap_counter += 1

        # 添加到初始化缓冲区
        self.init_kf_buffer.extend(new_keyframes)
        self.init_imu_factors_buffer.extend(imu_factors)

        print(f"【System Init】: Collecting data... KFs in buffer: {len(self.init_kf_buffer)}/{self.init_window_size}")

        if len(self.init_kf_buffer) >= self.init_window_size and self.check_motion_excitement():
            print("【System Init】: Motion excitement detected. Starting System initialization.")

            # 获取重力模长
            gravity_magnitude = self.config.get('gravity', 9.81)

            # 获取相机外参
            T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
            T_bc = np.asarray(T_bc_raw).reshape(4, 4)

            # 初始化
            flag, scale, gravity_c0, initial_gyro_bias, initial_velocities = VIOInitializer.initialize(
                self.init_kf_buffer, self.init_imu_factors_buffer, self.imu_processor,
                gravity_magnitude, T_bc
                )

            if flag:
                initial_bias_obj = gtsam.imuBias.ConstantBias(np.zeros(3), initial_gyro_bias)

                self.imu_processor.update_bias(initial_bias_obj)

                self.is_initialized = True
                print("【System Init】: System initialization successful!")
                
                # repropagated_imu_factors = VIOInitializer.repropagate_imu_factors(self.init_imu_factors_buffer, self.imu_processor, initial_gyro_bias)

                # 第一次初始化结果送入后端优化
                if self.backend:
                    imu_factors_copy = self.copy_imu_factors_for_backend(self.init_imu_factors_buffer)
                    self.backend.add_new_measurements(
                        self.init_kf_buffer, 
                        imu_factors_copy,
                        initial_velocities.copy(), 
                        initial_bias_obj
                    )

            else:
                print("【System Init】: Failed to initialize! Slide the window to try again...")
                # If init fails, slide the window to try again
                self.init_kf_buffer.pop(0)
                self.init_imu_factors_buffer.pop(0)
                self.is_initialized = False


    def process_package_data(self, package):
        """
            Args:
                pred_dict (dict):
                {
                    "images": (S, 3, H, W)   - Input images,
                    "world_points": (S, H, W, 3),
                    "world_points_conf": (S, H, W),
                    "depth": (S, H, W, 1),
                    "depth_conf": (S, H, W),
                    "extrinsic": (S, 3, 4),
                    "intrinsic": (S, 3, 3),
                }
        """

        visual_predictions = package['visual_predictions']
        imu_measurements = package['imu_measurements']

        if not visual_predictions or not imu_measurements:
            print("No visual predictions or imu measurements")
            return

        # 对IMU量测值进行预积分，注意这里使用后端优化最新的bias
        imu_factors = self.process_imu_data(imu_measurements, use_latest_bias=True)

        # 获取上一个submap用于对齐到全局Pose
        prev_submap = self.submaps.get(self.submap_counter - 1)

        # 创建新KF集
        new_keyframes, scale = self.create_and_align_keyframes(visual_predictions, prev_submap)

        if not new_keyframes:
            print("【System】: No new keyframes to process in this batch (likely all were overlapping).")
            return

        # 创建新submap
        new_submap = Submap(submap_id=self.submap_counter)
        new_submap.set_scale(scale)

        # 将新KF添加到global_map和submap
        for kf in new_keyframes:
            self.global_map.add_keyframe(kf)
            new_submap.add_keyframe(kf)

        # 添加submap并更新指针
        self.global_map.add_submap(new_submap)
        self.submaps[new_submap.id] = new_submap
        self.submap_counter += 1

        # 将新KF集和IMU因子发送到backend，触发后端优化
        print("【System】: Triggering backend optimization.")
        if self.backend and imu_factors:
            imu_factors_copy = self.copy_imu_factors_for_backend(imu_factors)
            self.backend.add_new_measurements(new_keyframes, imu_factors_copy)

        # 只发送增量数据
        # if not self.viewer_queue.full():
        #     # 从新子图中获取对齐后的点云和位姿
        #     new_points, new_colors, _ = new_submap.get_global_point_clouds()
        #     new_poses = new_submap.get_global_poses()
            
        #     # 打包成一个增量数据包
        #     incremental_data = {
        #         'points': new_points,
        #         'colors': new_colors,
        #         'poses': new_poses
        #     }
        #     self.viewer_queue.put(incremental_data)
        
    def create_and_align_keyframes(self, predictions, prev_submap):        
        # 读取预测结果
        timestamps = predictions.get("timestamps")
        images = predictions["images"]
        depth_maps = predictions['depth']
        depth_conf = predictions['depth_conf']
        cam_extrinsics = predictions["extrinsic"]
        cam_intrinsics = predictions['intrinsic']

        # 处理相机相关参数，这里实际上是对相机外参T_cam_world的逆变换->T_world_cam(world就是submap的第一帧坐标系)，同时处理为4x4矩阵T_first_last
        local_poses = closed_form_inverse_se3(cam_extrinsics)

        # 根据深度图恢复点云
        local_point_clouds = unproject_depth_map_to_point_map(depth_maps, cam_extrinsics, cam_intrinsics)
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # Convert images from (S, 3, H, W) to (S, H, W, 3)

        # 对齐到global坐标系的pose(T_c0_cam)，注意c0是第一帧的相机坐标系
        if prev_submap is None:
            scale = 1.0
            global_poses = local_poses
        else:
            scale, global_poses = align_submaps(prev_submap, local_poses, local_point_clouds, colors, depth_conf, self.confidence_threshold)

        new_keyframes = []
        for i in range(len(timestamps)):
            ts = timestamps[i]
            if self.global_map.get_keyframe(ts) is not None:
                continue

            # 创建一个新keyframe对象
            new_kf = KeyFrame()
            new_kf.set_timestamp(timestamps[i])
            new_kf.set_image(images[i])
            new_kf.set_local_pose(local_poses[i])
            new_kf.set_global_pose(global_poses[i])
            new_kf.set_point_cloud(local_point_clouds[i], colors[i])
            new_kf.set_confidence(depth_conf[i])

            new_keyframes.append(new_kf)
        
        return new_keyframes, scale

    def copy_imu_factors_for_backend(self, imu_factors):
        if not imu_factors:
            return []
        
        imu_factors_copy = []
        for factor_info in imu_factors:
            factor_copy = factor_info.copy()
            original_pim = factor_info.get('imu_preintegration')

            if original_pim:
                factor_copy['imu_preintegration'] = self.imu_processor.pre_integration(
                    factor_info['measurements'],
                    factor_info['start_kf_timestamp'],
                    factor_info['end_kf_timestamp'],
                    override_bias=original_pim.biasHat() # Use the exact same bias from the original
                )
            imu_factors_copy.append(factor_copy)
        return imu_factors_copy

    def process_imu_data(self, imu_measurements, use_latest_bias=False):
        imu_factors = []

        if use_latest_bias:
            latest_bias = self.backend.get_latest_bias()
            if latest_bias:
                self.imu_processor.update_bias(latest_bias)

        for imu_measurement in imu_measurements:
            start_kf_timestamp = imu_measurement['start_kf_timestamp']
            end_kf_timestamp = imu_measurement['end_kf_timestamp']
            measurements = imu_measurement['measurements']
            imu_preintegration = self.imu_processor.pre_integration(measurements, start_kf_timestamp, end_kf_timestamp)

            if imu_preintegration:
                factor_info = {
                    'start_kf_timestamp': start_kf_timestamp,
                    'end_kf_timestamp': end_kf_timestamp,
                    'measurements': measurements, # Keep raw data for re-propagation
                    'imu_preintegration': imu_preintegration # Store pre-integration result for backend
                }
                imu_factors.append(factor_info)

        return imu_factors

    def check_motion_excitement(self):
        if len(self.init_imu_factors_buffer) < 2:
            print(f"【System Init】: Not enough IMU factors for excitation check")
            return False

        # 按VINS-Fusion逻辑，计算IMU标准差
        accel_measurements = []

        for imu_factor in self.init_imu_factors_buffer:
            pim = imu_factor['imu_preintegration']
            delta_v = pim.deltaVij()
            accel_measurements.append(delta_v)

        if len(accel_measurements) > 1:
            accel_array = np.array(accel_measurements)
            accel_std = np.std(accel_array)

            if accel_std > 0.25:
                return True

        print("【System Init】: Not enough motion excitement")
        # return True
        return False