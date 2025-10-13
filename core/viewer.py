# utils/viewer3d.py
import open3d as o3d
import numpy as np

class Viewer3D:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.point_cloud = o3d.geometry.PointCloud()
        self.is_initialized = False

    def update(self, landmarks, poses):
        # 更新点云
        points = np.array(list(landmarks.values()))
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        
        if not self.is_initialized:
            self.vis.add_geometry(self.point_cloud)
            self.is_initialized = True
        else:
            self.vis.update_geometry(self.point_cloud)

        # 移除旧的相机位姿
        self.vis.clear_geometries()
        self.vis.add_geometry(self.point_cloud)

        # 添加新的相机位姿
        for kf_id, pose in poses.items():
            # 创建坐标系来表示相机位姿
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(pose) # 应用 T_world_cam
            self.vis.add_geometry(coord_frame)

        self.vis.poll_events()
        self.vis.update_renderer()