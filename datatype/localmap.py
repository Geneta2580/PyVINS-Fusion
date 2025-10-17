from collections import deque
from datatype.landmark import Landmark, LandmarkStatus
import numpy as np

class LocalMap:
    def __init__(self, config):
        self.config = config
        self.max_keyframes = self.config.get('window_size', 10)

        # 使用字典来存储，方便通过ID快速访问
        self.keyframes = {}  # {kf_id: KeyFrame_Object}
        self.landmarks = {}  # {lm_id: Landmark_Object}

    def add_keyframe(self, kf):
        self.keyframes[kf.get_id()] = kf

        # 更新Landmark的观测信息，或创建新的Landmark
        for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
            if lm_id in self.landmarks:
                self.landmarks[lm_id].add_observation(kf.get_id(), pt_2d)
            else:
                # 这是一个全新的Landmark
                new_lm = Landmark(lm_id, kf.get_id(), pt_2d)
                self.landmarks[lm_id] = new_lm
        
        # 维护滑动窗口，剔除最老的关键帧
        if len(self.keyframes) > self.max_keyframes:
            # 找到ID最小的关键帧
            oldest_kf_id = min(self.keyframes.keys())
            print(f"【LocalMap】: Sliding window is full. Removing oldest KeyFrame {oldest_kf_id}.")
            del self.keyframes[oldest_kf_id]

            # 关键帧被移除后，需要清理一下不再被观测的路标点
            self.prune_stale_landmarks()

    def prune_stale_landmarks(self):
        active_landmark_ids = set()
        for kf in self.keyframes.values():
            active_landmark_ids.update(kf.get_visual_feature_ids())

        stale_ids = [lm_id for lm_id in self.landmarks if lm_id not in active_landmark_ids]
        
        if stale_ids:
            print(f"【LocalMap】: Pruning {len(stale_ids)} stale landmarks.")
            print(f"【LocalMap】: Stale landmarks: {stale_ids}")
            for lm_id in stale_ids:
                del self.landmarks[lm_id]

    def get_active_keyframes(self):
        # 按ID排序后返回，确保顺序
        return sorted(self.keyframes.values(), key=lambda kf: kf.get_id())
    
    def get_active_landmarks(self):
        return {lm.id: lm.position_3d for lm in self.landmarks.values() if lm.status == LandmarkStatus.TRIANGULATED}

    def get_candidate_landmarks(self):
        return [lm for lm in self.landmarks.values() if lm.status == LandmarkStatus.CANDIDATE]

    def check_landmark_health(self, landmark_id, min_baseline_m=0.2):
        # 1. 找到所有观测到此路标点的活动关键帧
        lm = self.landmarks.get(landmark_id)
        if not lm:
            return False

        observing_kf_ids = lm.get_observing_kf_ids()
        witness_kfs = [self.keyframes[kf_id] for kf_id in observing_kf_ids if kf_id in self.keyframes]

        # 2. 至少需要2个观测帧才能计算基线
        if len(witness_kfs) < 2:
            return False
            
        positions = []
        for kf in witness_kfs:
            pose = kf.get_global_pose()
            positions.append(pose[:3, 3])

        # 4. 确保我们成功获取了至少2个位姿
        if len(positions) < 2:
            return False
            
        # 5. 计算观测基线（所有观测位置点构成的包围盒的对角线长度）
        positions = np.array(positions)
        baseline = np.linalg.norm(np.ptp(positions, axis=0))

        # 6. 与阈值比较
        if baseline < min_baseline_m:
            # print(f"【Health Check】: Landmark {landmark_id} is UNHEALTHY, baseline: {baseline:.4f}m")
            return False
        
        # print(f"【Health Check】: Landmark {landmark_id} is HEALTHY, baseline: {baseline:.4f}m")
        return True