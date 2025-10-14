import open3d as o3d
import numpy as np
import threading
import queue

class Viewer3D(threading.Thread):
    def __init__(self, viewer_queue):
        super().__init__(daemon=True)
        self.viewer_queue = viewer_queue
        self.is_running = False
        self.lock = threading.Lock()
        
        # 数据缓存
        self.landmarks = {}
        self.poses = {}
        
        # Open3D state
        self.vis = None
        self.point_cloud = o3d.geometry.PointCloud()
        self.camera_geometries = {}
        
        # --- 【关键新增】 ---
        # 用于判断是否是第一次渲染场景，以便自动调整视角
        self.scene_initialized = False 
        # --------------------

    def run(self):
        self.is_running = True
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("PyVINS-Fusion Viewer")
        self.vis.add_geometry(self.point_cloud)
        
        print("【Viewer】 thread started. Waiting for data...")

        while self.is_running:
            try:
                data = self.viewer_queue.get(timeout=0.01) 
                if data is None: break
                
                print("【Viewer】: Received new data, caching for update.")
                with self.lock:
                    if 'landmarks' in data:
                        self.landmarks.update(data['landmarks'])
                    if 'poses' in data:
                        self.poses.update(data['poses'])
                
            except queue.Empty:
                pass
            
            self._render_current_scene()
            
            if not self.vis.poll_events():
                break

        self.is_running = False
        if self.vis:
            self.vis.destroy_window()
        print("【Viewer】 thread has finished.")

    def _render_current_scene(self):
        with self.lock:
            # 更新点云
            if self.landmarks:
                points = np.array(list(self.landmarks.values()))
                self.point_cloud.points = o3d.utility.Vector3dVector(points)
                self.vis.update_geometry(self.point_cloud)

            # 更新相机位姿
            for geom in self.camera_geometries.values():
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.camera_geometries.clear()
            
            for kf_id, pose_matrix in self.poses.items():
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                coord_frame.transform(pose_matrix)
                self.vis.add_geometry(coord_frame, reset_bounding_box=False)
                self.camera_geometries[kf_id] = coord_frame
        
        # --- 【关键修改】 ---
        # 如果是第一次有数据被渲染，则自动调整视角
        if not self.scene_initialized and (self.landmarks or self.poses):
            self.vis.reset_view_point(True)
            self.scene_initialized = True
        # --------------------

        self.vis.update_renderer()

    def shutdown(self):
        if self.is_running:
            self.viewer_queue.put(None)
        self.is_running = False