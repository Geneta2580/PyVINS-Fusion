from pyexpat import features
import numpy as np
import threading
from collections import deque
from enum import Enum, auto

from utils.dataloader import ImuMeasurement
from core.visual_process import VisualProcessor
from core.imu_process import IMUProcessor

class FeatureTrackerState(Enum):
    IDLE = auto()
    AWAITING_LAST_KF_INTERPOLATION = auto()

class FeatureTracker:
    def __init__(self, config, dataloader, output_queue):
        self.config = config
        self.dataloader = dataloader
        self.output_queue = output_queue

        self.imu_buffer = deque()
        self.last_processed_kf_timestamp = None

        self.state = FeatureTrackerState.IDLE
        self.temporary_data = {}

        # Threading control
        self.is_running = False
        self.thread = threading.Thread(target=self.run, daemon=True)

        # 视觉处理模块
        self.visual_processor = VisualProcessor(config)

    def start(self):
        self.is_running = True
        self.thread.start()

    def shutdown(self):
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join()
        print("Visual Feature Tracker shut down.")

    def run(self):
        print("Visual Feature Tracker thread started.")
        for i, (timestamp, event_type, data) in enumerate(self.dataloader):
            if self.config.get('dataset_type', 'euroc') == 'euroc':
                timestamp = timestamp * 1e-9 # euroc数据集单位转换为s

            # 如果frontend被关闭，则退出循环
            if not self.is_running:
                break

            # 处理IMU数据
            if event_type == 'IMU':
                # 注意角速度在前，加速度在后
                data = ImuMeasurement(gyro = data[0:3], accel = data[3:6])
                imu_tuple = (timestamp, data)
                # 存入buffer
                self.imu_buffer.append(imu_tuple)
                
                # 若处于等待最后一个KF的IMU插值数据状态，则处理IMU数据
                if self.state == FeatureTrackerState.AWAITING_LAST_KF_INTERPOLATION:
                    kf_timestamps = self.temporary_data['visuals']['timestamps']
                    imu_measurements_list = []
                    for current_kf_timestamp in kf_timestamps:

                        # 重合/旧的帧，只更新时间戳指针
                        if current_kf_timestamp <= self.last_processed_kf_timestamp:                            
                            continue

                        start_time = self.last_processed_kf_timestamp
                        end_time = current_kf_timestamp

                        # 从buffer中获取历史IMU数据，同时清空历史IMU数据
                        measurements, self.imu_buffer = IMUProcessor.get_imu_interval_with_interpolation(self.imu_buffer, end_time)

                        if measurements:
                            imu_measurements = {
                                'start_kf_timestamp': start_time,
                                'end_kf_timestamp': end_time,
                                'measurements': measurements,
                            }
                            imu_measurements_list.append(imu_measurements)

                        self.last_processed_kf_timestamp = current_kf_timestamp

                    if imu_measurements_list:
                        output_data = {
                            'visual_predictions': self.temporary_data['visuals'],
                            'imu_measurements': imu_measurements_list,
                        }

                        self.output_queue.put(output_data)
                        self.temporary_data.clear()
                        self.state = FeatureTrackerState.IDLE

            # 处理图像数据
            elif event_type == 'IMAGE':
                image_data, image_path = data

                # 光流追踪特征点
                good_features_num, mean_parallax = self.visual_processor.track_features(image_data, self.config.get('visualize', False))                 

                

                # 处理关键帧
                visual_predictions = self.visual_processor.process_new_keyframe(image_path, timestamp)
                
                # 第一帧以及非关键帧都没有visual_predictions结果
                if visual_predictions is not None:
                    new_kf_timestamps = visual_predictions['timestamps']

                    # 第一个KF，清除所有第一个KF前的所有IMU数据
                    if self.last_processed_kf_timestamp is None:
                        self.last_processed_kf_timestamp = new_kf_timestamps[0]
                        _, self.imu_buffer = IMUProcessor.get_imu_interval_with_interpolation(self.imu_buffer, self.last_processed_kf_timestamp)

                    # 预测结果存入临时变量，等待最后一个KF的IMU插值数据
                    self.temporary_data['visuals'] = visual_predictions
                    self.state = FeatureTrackerState.AWAITING_LAST_KF_INTERPOLATION
        
        # 从数据循环中跳出，表示程序需要结束
        self.output_queue.put(None) 
        self.is_running = False
        print("Visual Feature Tracker has finished processing all data.")