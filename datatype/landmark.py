from enum import Enum
import numpy as np

class LandmarkStatus(Enum):
    CANDIDATE = 0  # 候选点，尚未被三角化
    TRIANGULATED = 1 # 已三角化，有3D位置

class Landmark:
    def __init__(self, landmark_id, first_kf_id, first_pt_2d):
        self.id = landmark_id
        
        # 初始状态为候选点，还没有3D位置
        self.status = LandmarkStatus.CANDIDATE
        self.position_3d = None
        
        # 记录所有的观测 {kf_id: pt_2d_coords}
        self.observations = {first_kf_id: first_pt_2d}

    def add_observation(self, kf_id, pt_2d):
        self.observations[kf_id] = pt_2d

    def remove_observation(self, kf_id):
        if kf_id in self.observations:
            del self.observations[kf_id]

    def get_observation_count(self):
        return len(self.observations)

    def get_observing_kf_ids(self):
        return self.observations.keys()

    def get_observation(self, kf_id):
        return self.observations[kf_id]

    def set_triangulated(self, position_3d):
        self.position_3d = position_3d
        self.status = LandmarkStatus.TRIANGULATED

    def is_ready_for_triangulation(self, keyframe_window, min_parallax):
        # 必须是候选点，且至少有三个观测
        if self.status != LandmarkStatus.CANDIDATE or self.get_observation_count() < 3:
            return False, None, None

        # 找到第一个和最后一个观测它的、且仍在滑动窗口内的关键帧
        obs_ids = list(self.observations.keys())
        first_kf_id = min(obs_ids)
        last_kf_id = max(obs_ids)

        first_kf = next((kf for kf in keyframe_window if kf.get_id() == first_kf_id), None)
        last_kf = next((kf for kf in keyframe_window if kf.get_id() == last_kf_id), None)

        if first_kf is None or last_kf is None or first_kf_id == last_kf_id:
            return False, None, None

        # 检查视差
        pt1 = self.observations[first_kf_id]
        pt2 = self.observations[last_kf_id]
        parallax = np.linalg.norm(pt1 - pt2)

        if parallax > min_parallax:
            return True, first_kf, last_kf
        else:
            # print(f"[Trace l{self.id}]: FAILED triangulation check. Parallax: {parallax:.2f}px")
            return False, None, None