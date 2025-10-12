import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Union

@dataclass
class ImuMeasurement:
    accel: Union[np.ndarray, list]
    gyro: Union[np.ndarray, list]

class UnifiedDataloader:
    def __init__(self, dataset_config: dict):
        self.base_path = Path(dataset_config['path'])
        self.dataset_type = dataset_config.get('dataset_type', 'euroc').lower()
        
        print(f"【Dataloader】: Initializing for dataset type: {self.dataset_type}")
        
        # 根据数据集类型设置路径
        if self.dataset_type == 'euroc':
            cam0_csv_path = self.base_path / 'image' / 'data.csv'
            imu0_csv_path = self.base_path / 'imu' / 'data.csv'
            self.cam_data_path = self.base_path / 'image' / 'data'
        elif self.dataset_type == 'tum':
            cam0_csv_path = self.base_path / 'image' / 'rgb.txt'
            self.cam_data_path = self.base_path / 'image'
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        try:
            # 读取相机数据
            if self.dataset_type == 'euroc':
                cam0_df = pd.read_csv(
                    cam0_csv_path,
                    header=0,
                    names=['timestamp', 'filename']
                )
            else:
                cam0_df = pd.read_csv(
                    cam0_csv_path,
                    comment='#',  
                    header=None,  
                    sep=' ',
                    names=['timestamp', 'filename']
                )
            
            print(f"【Dataloader】: Loaded {len(cam0_df)} image records")
            
            # --- 根据配置决定是否读取IMU数据 ---
            if self.dataset_type == 'euroc':
                # EuRoC数据集：读取IMU数据
                try:
                    imu0_df = pd.read_csv(
                        imu0_csv_path,
                        header=0,
                        names=['timestamp', 'w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']
                    )
                    print(f"【Dataloader】: Loaded {len(imu0_df)} IMU records")
                    
                    # 为数据添加类型标签
                    cam0_df['type'] = 'IMAGE'
                    imu0_df['type'] = 'IMU'
                    
                    # 合并相机和IMU数据
                    self.unified_data = pd.concat([cam0_df, imu0_df])
                    
                except FileNotFoundError:
                    print(f"【Warning】: IMU data file not found at {imu0_csv_path}")
                    print(f"【Info】: Falling back to vision-only mode")
                    cam0_df['type'] = 'IMAGE'
                    self.unified_data = cam0_df
                    
            else:  # dataset_type == 'tum' 或其他纯视觉数据集
                # TUM数据集：只使用相机数据
                print(f"【Dataloader】: IMU data disabled for {self.dataset_type} dataset")
                cam0_df['type'] = 'IMAGE'
                self.unified_data = cam0_df
            
        except FileNotFoundError as e:
            print(f"Error: Could not find camera data file at {cam0_csv_path}")
            raise e

        # 按时间戳排序
        self.unified_data.sort_values(by='timestamp', inplace=True, ignore_index=True)
        
        self.current_idx = 0
        print(f"【Dataloader】: Dataloader initialized with {len(self.unified_data)} total events")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.unified_data):
            raise StopIteration

        event = self.unified_data.iloc[self.current_idx]
        self.current_idx += 1
        
        # 根据事件类型，准备并返回相应的数据
        event_type = event['type']
        timestamp = event['timestamp']

        if event_type == 'IMAGE':
            img_filename = event['filename']
            img_path = self.cam_data_path / img_filename
            
            # 检查图像文件是否存在
            if not img_path.exists():
                print(f"【Warning】: Image file not found: {img_path}")
                return self.__next__()  # 跳过这一帧
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"【Warning】: Failed to load image: {img_path}")
                return self.__next__()  # 跳过这一帧
            
            return timestamp, event_type, (image, str(img_path))
        
        elif event_type == 'IMU':
            # 返回IMU数据 (w_x, w_y, w_z, a_x, a_y, a_z)
            imu_data = event[['w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']].values
            return timestamp, event_type, imu_data