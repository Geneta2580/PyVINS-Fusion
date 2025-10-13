import numpy as np
from utils.geometry import filter_point_cloud

class KeyFrame:
    def __init__(self):
        self.id = -1
        self.timestamp = None  # Timestamps for each frame
        self.image = None  # Image tensors from the keyframe buffer (3, H, W)
        self.local_pose = None  # Camera extrinsics for each frame (4, 4)
        self.global_pose = None  # Global poses for each frame (4, 4)，global_pose在初始化前是T_c0_ci，初始化后是T_w_ci
        self.point_cloud = None # Dict to store point clouds (H, W, 3)
        self.color = None  # Dict to store colors (H, W, 3)
        self.confidence = None  # Dict to store confidence (H, W)

        self.imu_measurements = None # IMU measurements for each frame (7)

    # 写入类信息(write)
    def set_id(self, id):
        self.id = id

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    def set_image(self, image):
        self.image = image
        
    def set_local_pose(self, local_pose):
        self.local_pose = local_pose

    def set_global_pose(self, global_pose):
        self.global_pose = global_pose

    def set_point_cloud(self, point_cloud, color):
        self.point_cloud = point_cloud
        self.color = color
        
    def set_confidence(self, confidence):
        self.confidence = confidence

    # 读取类信息(read)
    def get_id(self):
        return self.id

    def get_timestamp(self):
        return self.timestamp
    
    def get_image(self):
        return self.image

    def get_local_pose(self):
        return self.local_pose

    def get_global_pose(self):
        return self.global_pose

    def get_point_cloud(self):
        return self.point_cloud
        
    def get_color(self):
        return self.color

    def get_confidence(self):
        return self.confidence

    def get_global_point_cloud(self, scale, conf_threshold):
        # 根据类内pose计算global点云，方便后续优化调整
        # 获取filter过后的点云
        filtered_point_cloud, filtered_color, filtered_confidence = filter_point_cloud(self.point_cloud, self.confidence, self.color, conf_threshold)
        
        # 统一尺寸
        point_clouds_scaled = filtered_point_cloud * scale
        points_h = np.hstack([point_clouds_scaled, np.ones((point_clouds_scaled.shape[0], 1))]) # 扩展点云为齐次坐标(4)
        points_transformed_h = (self.global_pose @ points_h.T).T # 变换到global的齐次坐标(4)
        curr_global_pcd = points_transformed_h[:, :3] / points_transformed_h[:, 3, np.newaxis] # 变换到世界坐标系(3)

        return curr_global_pcd, filtered_color, filtered_confidence
    