import numpy as np
import threading

class GlobalMap:
    """
    Manages the global map, aggregating point clouds and poses from all submaps.
    """
    def __init__(self):
        self.keyframes = {}
        self.submap = {}

        self.next_kf_id = 0
        self.lock = threading.Lock() # 访问global_map的锁

    def add_keyframe(self, keyframe):
        with self.lock:
            # 避免重复添加
            if keyframe.get_timestamp() in self.keyframes:
                return

            # 分配KF全局id
            kf_id = self.next_kf_id
            keyframe.set_id(kf_id)

            # 时间戳作为查询key，查询哈希更快
            self.keyframes[keyframe.get_timestamp()] = keyframe
            self.next_kf_id += 1

    def add_submap(self, submap):
        with self.lock:
            self.submap[submap.id] = submap

    def get_keyframe(self, timestamp):
        with self.lock:
            return self.keyframes.get(timestamp)

    def get_all_keyframes(self):
        with self.lock:
            return list(self.keyframes.values())

    def get_submap(self, submap_id):
        with self.lock:
            return self.submap.get(submap_id)

    def get_all_submaps(self):
        with self.lock:
            return list(self.submap.values())
    
    def get_global_points_and_colors(self, conf_threshold=0.9): # 注意这里conf_threshold知识用来筛选显示，不影响pose计算
        all_points = []
        all_colors = []
        
        with self.lock:
            submaps_to_process = list(self.submaps.values())


        for submap in submaps_to_process:
            points, colors, _ = submap.get_global_point_clouds(conf_threshold)
            if points is not None and len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)

        if not all_points:
            return np.empty((0, 3)), np.empty((0, 3))

        return np.vstack(all_points), np.vstack(all_colors)
    
    def get_global_poses(self):
        all_poses = []
        with self.lock:
            keyframes_sorted = sorted(self.keyframes.values(), key=lambda kf: kf.get_timestamp())
        
        for kf in keyframes_sorted:
            pose = kf.get_global_pose()
            if pose is not None:
                all_poses.append(pose)
        
        return all_poses

    def update_keyframe_pose(self, timestamp, optimized_pose_np):
        with self.lock:
            kf = self.keyframes.get(timestamp)
            if kf:
                kf.set_global_pose(optimized_pose_np)