import numpy as np
import threading
from collections import deque
from enum import Enum, auto

from .visual_process import VisualProcessor
from .imu_process import IMUProcessor
from utils.dataloader import ImuMeasurement

class FrontendState(Enum):
    IDLE = auto()
    AWAITING_LAST_KF_INTERPOLATION = auto()

class Frontend:
    def __init__(self, checkpoint_path, config, dataloader, imu_processor, output_queue):
        self.config = config
        self.dataloader = dataloader
        self.output_queue = output_queue

        self.imu_buffer = deque()
        self.last_processed_kf_timestamp = None

        self.state = FrontendState.IDLE
        self.temporary_data = {}

        self.visual_processor = VisualProcessor(checkpoint_path, config)
        self.imu_processor = imu_processor

        # Threading control
        self.is_running = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.is_running = True
        self.thread.start()

    def shutdown(self):
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join()
        print("Visual Frontend shut down.")

    def run(self):
        print("Frontend thread started.")
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
                # self.imu_processor.fast_integration(data)
                
                # 若处于等待最后一个KF的IMU插值数据状态，则处理IMU数据
                if self.state == FrontendState.AWAITING_LAST_KF_INTERPOLATION:
                    kf_timestamps = self.temporary_data['visuals']['timestamps']
                    imu_measurements_list = []
                    for current_kf_timestamp in kf_timestamps:

                        # 重合/旧的帧，只更新时间戳指针
                        if current_kf_timestamp <= self.last_processed_kf_timestamp:                            
                            continue

                        start_time = self.last_processed_kf_timestamp
                        end_time = current_kf_timestamp

                        # 从buffer中获取历史IMU数据，同时清空历史IMU数据
                        measurements, self.imu_buffer = self.imu_processor.get_imu_interval_with_interpolation(self.imu_buffer, end_time)

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
                        self.state = FrontendState.IDLE

            # 处理图像数据
            elif event_type == 'IMAGE':
                image_data, image_path = data

                # 判断关键帧
                is_keyframe = self.visual_processor.is_keyframe(image_data, timestamp)

                # 非关键帧直接跳过
                if not is_keyframe:
                    continue

                # 处理关键帧
                visual_predictions = self.visual_processor.process_new_keyframe(image_path, timestamp)
                
                # 第一帧以及非关键帧都没有visual_predictions结果
                if visual_predictions is not None:
                    new_kf_timestamps = visual_predictions['timestamps']

                    # 第一个KF，清除所有第一个KF前的所有IMU数据
                    if self.last_processed_kf_timestamp is None:
                        self.last_processed_kf_timestamp = new_kf_timestamps[0]
                        _, self.imu_buffer = self.imu_processor.get_imu_interval_with_interpolation(self.imu_buffer, self.last_processed_kf_timestamp)

                    # 预测结果存入临时变量，等待最后一个KF的IMU插值数据
                    self.temporary_data['visuals'] = visual_predictions
                    self.state = FrontendState.AWAITING_LAST_KF_INTERPOLATION

                    # test纯视觉
                    # output_data = {
                    #     'visual_predictions': self.temporary_data['visuals'],
                    #     'imu_factors': imu_factors_list,
                    # }

                    # self.output_queue.put(output_data)
                    # self.temporary_data.clear()
                    # test纯视觉
        
        # 从数据循环中跳出，表示程序需要结束
        self.output_queue.put(None) 
        self.is_running = False
        print("Frontend has finished processing all data.")