from collections import deque
from datatype.landmark import Landmark, LandmarkStatus
import numpy as np
import cv2
import gtsam
import time

class LocalMap:
    def __init__(self, config):
        self.config = config
        self.max_keyframes = self.config.get('window_size', 10)

        self.cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)

        # 使用字典来存储，方便通过ID快速访问
        self.keyframes = {}  # {kf_id: KeyFrame_Object}
        self.landmarks = {}  # {lm_id: Landmark_Object}

    def add_keyframe(self, kf):
        self.keyframes[kf.get_id()] = kf

        suspect_lm_id = 14815 # <--- 设置我们要追踪的目标

        # 更新Landmark的观测信息，或创建新的Landmark，创建后默认为CANDIDATE
        # DEBUG
        for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
            if lm_id in self.landmarks:
                self.landmarks[lm_id].add_observation(kf.get_id(), pt_2d)
                if lm_id == suspect_lm_id:
                    print(f"🕵️‍ [Trace l{suspect_lm_id}]: OBSERVED by new KF {kf.get_id()}. Total observations: {self.landmarks[lm_id].get_observation_count()}")
            else:
                new_lm = Landmark(lm_id, kf.get_id(), pt_2d)
                self.landmarks[lm_id] = new_lm
                if lm_id == suspect_lm_id:
                    print(f"🕵️‍ [Trace l{suspect_lm_id}]: --- BORN! First seen in KF {kf.get_id()} ---")
        # DEBUG
        
        # 维护滑动窗口，剔除最老的关键帧
        if len(self.keyframes) > self.max_keyframes:
            # 找到ID最小的关键帧
            oldest_kf = min(self.keyframes.values(), key=lambda kf: kf.get_timestamp())
            oldest_kf_id = oldest_kf.get_id()
            print(f"【LocalMap】: Sliding window is full. Removing oldest KeyFrame {oldest_kf_id}.")
            del self.keyframes[oldest_kf_id]

            for landmark in self.landmarks.values():
                landmark.remove_observation(oldest_kf_id)

            # 关键帧被移除后，需要清理一下不再被观测的路标点
            stale_lm_ids = self.prune_stale_landmarks()
            return stale_lm_ids
        
        return None

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
            
            return stale_ids
        
        return None

    def get_active_keyframes(self):
        # 按ID排序后返回，确保顺序
        return sorted(self.keyframes.values(), key=lambda kf: kf.get_id())
    
    def get_active_landmarks(self):
        return {lm.id: lm.position_3d for lm in self.landmarks.values() if lm.status == LandmarkStatus.TRIANGULATED}

    def get_candidate_landmarks(self):
        return [lm for lm in self.landmarks.values() if lm.status == LandmarkStatus.CANDIDATE]

    def check_landmark_health(self, landmark_id, candidate_position_3d=None, min_parallax_angle_deg=3.0):
        lm = self.landmarks.get(landmark_id)
        # 必须是已三角化的点才有3D位置
        if not lm:
            return False

        # 对于还没有确认三角化的点，使用候选位置
        if candidate_position_3d is not None:
            landmark_pos = candidate_position_3d
        # 对于已经三角化的点，使用三角化后的位置
        elif lm.status == LandmarkStatus.TRIANGULATED and lm.position_3d is not None:
            landmark_pos = lm.position_3d
        else:
            return False

        observing_kf_ids = lm.get_observing_kf_ids()
        witness_kfs = [self.keyframes[kf_id] for kf_id in observing_kf_ids if kf_id in self.keyframes]

        # 至少需要3个观测帧
        if len(witness_kfs) < 3:
            return False
            
        positions = []
        for kf in witness_kfs:
            pose = kf.get_global_pose()
            if pose is not None:
                positions.append(pose[:3, 3])

        if len(positions) < 3:
            return False
            
        positions = np.array(positions)

        # 计算观测基线
        baseline = np.linalg.norm(np.ptp(positions, axis=0))

        # # 基线太短，排除
        # if baseline < 0.05:
        #     print(f"【Health Check】: Landmark {lm.id} failed baseline check. Baseline: {baseline:.4f}m")
        #     return False

        # 计算路标点到观测中心的大致深度
        avg_cam_pos = np.mean(positions, axis=0) # 观测中心
        depth = np.linalg.norm(landmark_pos - avg_cam_pos)

        # 避免除以零
        if depth < 1e-6:
            return False
        
        # 检查基线与深度的比值（近似于 2 * tan(parallax_angle / 2)）
        # 一个小的角度，tan(theta)约等于theta（弧度）
        ratio = baseline / depth
        threshold = np.deg2rad(min_parallax_angle_deg)

        if ratio < threshold:
            return False

        # 检查重投影误差和深度
        max_reprojection_error = 5.0 # px

        for kf in witness_kfs:
            pose = kf.get_global_pose()
            if pose is None: continue

            T_cam_world = np.linalg.inv(pose)
            point_in_cam_homo = T_cam_world @ np.append(landmark_pos, 1.0)
            
            # 深度必须为正
            depth = point_in_cam_homo[2] / point_in_cam_homo[3]
            if depth <= 0.1 or depth > 400.0:
                print(f"【Health Check】: Landmark {lm.id} failed cheirality in KF {kf.get_id()}. Depth: {depth:.4f}m")
                return False

            # rvec, _ = cv2.Rodrigues(T_cam_world[:3,:3])
            # tvec = T_cam_world[:3,3]
            # reprojected_pt, _ = cv2.projectPoints(landmark_pos.reshape(1,1,3), rvec, tvec, self.cam_intrinsics, None)
            # reproj_error = np.linalg.norm(reprojected_pt.flatten() - lm.observations[kf.get_id()])
            # if reproj_error > max_reprojection_error:
            #     print(f"【Health Check】: Landmark {lm.id} failed reprojection in KF {kf.get_id()}. Error: {reproj_error:.2f}px")
            #     return False

        if landmark_id == 14815: # 您可以修改为您想追踪的任何ID
            is_healthy = ratio >= threshold # 重新计算一下最终结果
            print("\n--- 🩺 Health Check Debug ---")
            print(f"  Landmark ID: {landmark_id}")
            print(f"  Observing KF IDs in window: {[kf.get_id() for kf in witness_kfs]}")
            print(f"  Baseline (B): {baseline:.4f} m")
            print(f"  Avg Depth (D): {depth:.4f} m")
            print(f"  Ratio (B/D): {ratio:.4f}")
            print(f"  Threshold (rad): {threshold:.4f}")
            print(f"  Result: {'HEALTHY (True)' if is_healthy else 'UNHEALTHY (False)'}")
            print("--- End of Health Check Debug ---\n")

        return True

    
    def check_landmark_depth(self, landmark_id, max_depth=400.0):
        lm = self.landmarks.get(landmark_id)
        # 必须是已三角化的点才有3D位置
        if not lm or lm.position_3d is None:
            return False

        observing_kf_ids = [kf_id for kf_id in lm.get_observing_kf_ids() if kf_id in self.keyframes]
        
        if len(observing_kf_ids) < 2:
            return False
            
        # 优化：只检查ID最小和最大的两个观测帧
        first_kf_id = min(observing_kf_ids)
        last_kf_id = max(observing_kf_ids)
        
        # 将要检查的关键帧限制在这两个极端
        kfs_to_check = [self.keyframes[first_kf_id]]
        if first_kf_id != last_kf_id:
            kfs_to_check.append(self.keyframes[last_kf_id])

        for kf in kfs_to_check:
            pose = kf.get_global_pose()
            if pose is None: continue

            T_cam_world = np.linalg.inv(pose)
            point_in_cam_homo = T_cam_world @ np.append(lm.position_3d, 1.0)
            
            # 检查深度是否为正且在合理范围内
            depth = point_in_cam_homo[2]
            if depth <= 0.1 or depth > max_depth:
                return False

        # if np.linalg.norm(lm.position_3d) > 20:
        #     return False

        return True