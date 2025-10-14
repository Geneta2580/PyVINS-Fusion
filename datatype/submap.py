import cv2 
import numpy as np
from .keyframe import KeyFrame

class Submap:
    def __init__(self, submap_id):
        self.id = submap_id
        self.keyframes = {}           
        self.scale = 1.0           # Scale factor for the submap relative to prior submap

    # 添加类内数据（write）
    def add_keyframe(self, keyframe: KeyFrame):
        self.keyframes[keyframe.get_id()] = keyframe

    def set_scale(self, scale):
        self.scale = scale

    # 访问类内数据（read）
    def get_id(self):
        return self.id

    def get_keyframe(self, keyframe_id):
        return self.keyframes.get(keyframe_id)

    def get_all_keyframes(self):
        return sorted(list(self.keyframes.values()), key=lambda kf: kf.get_timestamp())

    def get_scale(self):
        return self.scale

    def get_global_point_clouds(self, conf_threshold: float) -> (np.ndarray, np.ndarray):
        all_points_global = []
        all_colors_global = []

        # 遍历submap中的每一个keyframe
        for kf in self.get_all_keyframes():
            # 调用KeyFrame自己的方法，并传入它所需要的 scale 和 conf_threshold
            points_global, colors = kf.get_global_point_cloud(
                scale=self.scale, 
                conf_threshold=conf_threshold
            )
            
            if len(points_global) > 0:
                all_points_global.append(points_global)
                all_colors_global.append(colors)

        if not all_points_global:
            return np.empty((0, 3)), np.empty((0, 3))

        # 聚合一个submap中所有keyframe的global点云
        return np.vstack(all_points_global), np.vstack(all_colors_global)