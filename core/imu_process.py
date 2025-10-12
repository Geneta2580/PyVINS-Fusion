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
        self.params.setIntegrationCovariance(np.eye(3) * 1e-8) # 预积分协方差，通常可以设一个很小的值

        self.params.setBiasAccCovariance(np.eye(3) * accel_bias_rw_sigma**2) # 加计零偏随机游走
        self.params.setBiasOmegaCovariance(np.eye(3) * gyro_bias_rw_sigma**2) # 陀螺零偏随机游走
        self.params.setBiasAccOmegaInit(np.eye(6) * 1e-4) # 加计和陀螺零偏初始协方差

        self.current_bias = gtsam.imuBias.ConstantBias()

    def get_imu_interval_with_interpolation(self, imu_buffer_deque: deque, end_time: float) -> Tuple[List[ImuData], deque]:
        measurements_to_process = []

        while len(imu_buffer_deque) > 0 and imu_buffer_deque[0][0] <= end_time:
            # 添加到IMU测量列表，同时从缓冲区中删除这些数据
            measurements_to_process.append(imu_buffer_deque.popleft()) 

        # 提取一个额外的IMU数据，用于插值
        if len(imu_buffer_deque) > 0:
            measurements_to_process.append(imu_buffer_deque.popleft())

        return measurements_to_process, imu_buffer_deque

    def update_bias(self, new_bias):
        self.current_bias = new_bias

    def imu_interpolation(self, meas_before: ImuData, meas_after: ImuData, target_ts: float):
        ts_before, data_before = meas_before
        ts_after, data_after = meas_after

        if ts_after == ts_before:
            return data_before.accel, data_before.gyro

        # 按时间比例线性插值，同步到图像帧时间
        ratio = (target_ts - ts_before) / (ts_after - ts_before)
        interp_accel = data_before.accel + (data_after.accel - data_before.accel) * ratio
        interp_gyro = data_before.gyro + (data_after.gyro - data_before.gyro) * ratio
        
        return interp_accel, interp_gyro

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

        # 最后一个IMU数据用于插值，同时将最后一个IMU数据从列表中移除
        interpolation_point = measurements.pop()
        # 修改后的最后一个IMU数据，用于插值
        last_integrated_point = measurements[-1]

        # 积分IMU数据
        last_timestamp = start_time
        for ts, data in measurements:
            dt = ts - last_timestamp # 注意这里的单位应该是s
            # print(f"【IMU_process】dt: {dt}")
            if dt > 0:
                preintegrated_measurements.integrateMeasurement(data.accel, data.gyro, dt)
            last_timestamp = ts

        if end_time > last_timestamp:
            # 对最后一个IMU数据进行插值并积分
            interp_accel, interp_gyro = self.imu_interpolation(last_integrated_point, interpolation_point, end_time)
            dt =  end_time - last_timestamp
            preintegrated_measurements.integrateMeasurement(interp_accel, interp_gyro, dt)

        return preintegrated_measurements