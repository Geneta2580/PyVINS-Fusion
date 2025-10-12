import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import multiprocessing as mp 
import queue
import traceback

class Viewer(mp.Process):
    """
    Manages the Open3D visualization window in a separate PROCESS.
    """
    def __init__(self, input_queue):
        super().__init__()
        self.input_queue = input_queue
        self.is_running = mp.Value('b', True)

    def run(self):
        """
        The main loop of the viewer process. All Open3D operations happen here.
        """
        print("[Viewer Process] run() method has been entered.")

        # --- 关键修正：在这里进行延迟导入 ---
        import open3d as o3d

        vis = None
        try:
            print("[Viewer Process] Attempting to create o3d.visualization.Visualizer()...")
            vis = o3d.visualization.Visualizer()
            print("[Viewer Process] Visualizer object created successfully.")

            print("[Viewer Process] Attempting to create window...")
            vis.create_window(window_name="iVGGT-SLAM Map Viewer", width=1280, height=720)
            print("[Viewer Process] Window created.")

            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            vis.add_geometry(origin_frame)

            # --- 数据模型：Viewer 进程自己维护的、不断增长的地图 ---
            map_pcd = o3d.geometry.PointCloud()
            
            # --- 渲染状态：记录当前显示在场景中的几何体 ---
            scene_pcd = o3d.geometry.PointCloud()
            scene_poses = []
            is_scene_initialized = False

            voxel_size = 0.02

            # --- 关键修正：在这里初始化内部缓冲区 ---
            data_to_process = None

            while self.is_running.value:
                # --- 阶段一：非阻塞式地检查新数据 ---
                # 这个阶段非常快，几乎不耗时
                try:
                    # 如果内部缓冲区是空的，才尝试从队列里拿一个新的
                    if data_to_process is None:
                        data_to_process = self.input_queue.get_nowait()
                except queue.Empty:
                    pass

                # --- 阶段二：如果缓冲区有数据，则处理它 ---
                # 这个阶段是耗时的，但它只在有新数据时执行
                if data_to_process:
                    # --- 开始计时 ---
                    t_start = time.time()

                    print("[Viewer Process] Processing new incremental data.")
                    # 1. 高效地更新数据模型
                    new_points = data_to_process.get('points')
                    new_colors = data_to_process.get('colors')
                    if new_points is not None and len(new_points) > 0:
                        new_pcd = o3d.geometry.PointCloud()
                        new_pcd.points = o3d.utility.Vector3dVector(new_points)
                        new_pcd.colors = o3d.utility.Vector3dVector(new_colors / 255.0)

                        map_pcd += new_pcd
                        map_pcd = map_pcd.voxel_down_sample(voxel_size)

                    # 2. --- 关键修正：正确的渲染逻辑 ---
                    # a. 从场景中移除旧的几何体
                    if is_scene_initialized:
                        vis.remove_geometry(scene_pcd, reset_bounding_box=False)
                        for pose_geom in scene_poses:
                            vis.remove_geometry(pose_geom, reset_bounding_box=False)
                        scene_poses.clear()
                    
                    # b. 用更新后的数据模型来创建新的场景几何体
                    scene_pcd.points = map_pcd.points
                    scene_pcd.colors = map_pcd.colors
                    
                    new_poses_data = data_to_process.get('poses', []) # 假设我们只显示最新的位姿
                    for pose in new_poses_data:
                        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                        cam_frame.transform(pose)
                        scene_poses.append(cam_frame)

                    # c. 将新的几何体添加到场景
                    vis.add_geometry(scene_pcd, reset_bounding_box=False)
                    for pose_geom in scene_poses:
                        vis.add_geometry(pose_geom, reset_bounding_box=False)
                    is_scene_initialized = True

                    # 处理完成后，清空缓冲区
                    data_to_process = None
                    
                    # --- 结束计时 ---
                    t_end = time.time()
                    print(f"[PROFILE_VIEWER] Update time: {(t_end - t_start) * 1000:.2f} ms")


                # --- 阶段三：无条件地、高频率地执行渲染和交互 ---
                # 这个阶段非常快，保证了流畅性
                if not vis.poll_events():
                    self.is_running.value = False
                    break
                vis.update_renderer()
                
                # 仅在无事可做时休眠（没有新数据需要检查，也没有数据正在处理）
                if self.input_queue.empty() and data_to_process is None:
                    time.sleep(0.01)

        except Exception as e:
            print(f"[Viewer Process] An error occurred: {e}")
            print(traceback.format_exc())
        finally:
            print("[Viewer Process] Shutting down.")
            if vis:
                vis.destroy_window()

    def shutdown(self):
        """Signals the viewer process to stop its run loop."""
        print("[Main Process] Signaling viewer to shut down.")
        self.is_running.value = False
