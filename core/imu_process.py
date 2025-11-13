import numpy as np
import gtsam
from collections import deque
from typing import List, Tuple

ImuData = Tuple[float, any]

class IMUProcessor:
    def __init__(self, config):
        self.g = config.get('gravity', 9.81)
    
        # 从config文件获取IMU参数
        accel_noise_sigma = config.get('accel_noise_sigma', 1e-2)
        gyro_noise_sigma = config.get('gyro_noise_sigma', 1e-3)
        accel_bias_rw_sigma = config.get('accel_bias_rw_sigma', 1e-4)
        gyro_bias_rw_sigma = config.get('gyro_bias_rw_sigma', 1e-5)
        
        # 传递到GTSAM参数
        self.params = gtsam.PreintegrationCombinedParams.MakeSharedU(self.g) # 重力补偿参数

        self.params.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2) # 加计协方差
        self.params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2) # 陀螺协方差
        self.params.setIntegrationCovariance(np.eye(3) * 1e-5) # 预积分协方差，通常可以设一个很小的值

        self.params.setBiasAccCovariance(np.eye(3) * accel_bias_rw_sigma**2) # 加计零偏随机游走
        self.params.setBiasOmegaCovariance(np.eye(3) * gyro_bias_rw_sigma**2) # 陀螺零偏随机游走
        # self.params.setBiasAccCovariance(np.eye(3) * 1e-3)  # 加计零偏初始协方差
        # self.params.setBiasOmegaCovariance(np.eye(3) * 1e-3) # 陀螺零偏初始协方差

        self.current_bias = gtsam.imuBias.ConstantBias()

    @staticmethod
    def get_imu_interval_with(imu_buffer_deque: deque, end_time: float) -> Tuple[List[ImuData], deque]:
        measurements_to_process = []

        while len(imu_buffer_deque) > 0 and imu_buffer_deque[0][0] <= end_time:
            # 添加到IMU测量列表，同时从缓冲区中删除这些数据
            measurements_to_process.append(imu_buffer_deque.popleft()) 

        return measurements_to_process, imu_buffer_deque

    def update_bias(self, new_bias):
        self.current_bias = new_bias

    def fast_integration(self, imu_data):
        return None
        # print(f"IMU data: {imu_data}")

    def pre_integration(self, measurements: List[ImuData], start_time: float, end_time: float, override_bias = None):

        if len(measurements) < 2:
            print("[Warning] Not enough IMU measurements to perform pre-integration.")
            return None

        if override_bias is not None:
            # print("【IMU_process】: Using override bias")
            current_bias = override_bias
        else:
            # 假设每次都从零偏置开始，在实际系统中，这里应该传入上一个关键帧优化后的偏置
            current_bias = self.current_bias

        preintegrated_measurements = gtsam.PreintegratedCombinedMeasurements(self.params, current_bias)

        # 逐段积分IMU数据
        last_timestamp = start_time
        for imu_data in measurements:
            timestamp, data = imu_data

            if timestamp >= last_timestamp:
                dt = timestamp - last_timestamp # 注意这里的单位应该是s
                if dt <=0:
                    continue

                preintegrated_measurements.integrateMeasurement(data.accel, data.gyro, dt)

                last_timestamp = timestamp

        # 对最后一段IMU进行积分
        final_dt = end_time - last_timestamp
        if final_dt > 0:
            last_accel = measurements[-1][1].accel
            last_gyro = measurements[-1][1].gyro
            preintegrated_measurements.integrateMeasurement(last_accel, last_gyro, final_dt)

        return preintegrated_measurements