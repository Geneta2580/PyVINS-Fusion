import numpy as np
from scipy.spatial.transform import Rotation
# import torch

def filter_point_cloud(point_cloud, color, confidence, conf_threshold):
    # 这里过mask之后不改变point_clouds、colors、confidence的形状，所以能完成分组访问的功能
    # 每张图的置信度滤出的结果不相同，因此不能合并成point_clouds_filtered
    if point_cloud is None or confidence is None or color is None:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    
    mask = confidence > conf_threshold
    point_cloud_filtered = point_cloud[mask] # (H, W, 3)->(K, 3)
    color_filtered = color[mask]
    confidence_filtered = confidence[mask]

    return point_cloud_filtered, color_filtered, confidence_filtered

def pose_matrix_to_tum_format(pose_matrix):
    """Converts a 4x4 SE(3) pose matrix to a TUM trajectory format string components."""
    t = pose_matrix[:3, 3]
    q = Rotation.from_matrix(pose_matrix[:3, :3]).as_quat() # q is [x, y, z, w]
    return t[0], t[1], t[2], q[0], q[1], q[2], q[3]

def align_submaps(prev_submap, current_local_poses, current_local_point_clouds, current_colors, current_confidences, conf_threshold):

    # 获取上一个submap的最后一帧信息
    prev_last_kf = prev_submap.get_all_keyframes()[-1]
    prev_last_global_pose = prev_last_kf.get_global_pose()
    prev_last_local_pose = prev_last_kf.get_local_pose()
    
    # 获取filter过后的点云
    prev_pcd_filtered, _, _ = filter_point_cloud(
        prev_last_kf.get_point_cloud(),
        prev_last_kf.get_color(),
        prev_last_kf.get_confidence(),
        conf_threshold
    )

    curr_pcd_filtered, _, _ = filter_point_cloud(
        current_local_point_clouds[0],
        current_colors[0],
        current_confidences[0],
        conf_threshold
    )

    # 确保两组点云的长度一致
    min_len = min(len(prev_pcd_filtered), len(curr_pcd_filtered))
    prev_pcd = prev_pcd_filtered[:min_len]
    curr_pcd = curr_pcd_filtered[:min_len]

    # 上一个submap最后一帧相对于第一帧的pose，T_prev0_prev2->T_prev2_prev0
    T_temp = prev_last_local_pose
    T_temp_inv = np.linalg.inv(T_temp)

    # 分子：将上一个submap最后一帧的点云转换到最后一帧的坐标系 分母：当前submap第一帧的点云 将点云统一变换到重叠帧坐标系下，计算尺度因子
    points_h = np.hstack([prev_pcd, np.ones((prev_pcd.shape[0], 1))])
    points_transformed_h = (T_temp_inv @ points_h.T).T
    prev_pcd_transformed = points_transformed_h[:, :3]
    numerator = np.linalg.norm(prev_pcd_transformed, axis=1) * prev_submap.get_scale()
    denominator = np.linalg.norm(curr_pcd, axis=1)

    # 添加数值健康检查
    valid_indices = (denominator > 1e-6)  # 避免除零
    if np.sum(valid_indices) < 10:
        print(f"【Warning】: Too few valid points for scale estimation in submap")
        scale_factor = 1.0  # 使用默认尺度
    else:
        scale_factor = np.mean(numerator[valid_indices] / denominator[valid_indices])
    
    # 添加尺度异常检测
    if scale_factor < 0.1 or scale_factor > 10.0:
        print(f"【Warning】: Abnormal scale factor {scale_factor:.6f} detected for submap")
        print(f"  Numerator stats: mean={np.mean(numerator):.6f}, std={np.std(numerator):.6f}")
        print(f"  Denominator stats: mean={np.mean(denominator):.6f}, std={np.std(denominator):.6f}")
        # 可以选择限制尺度因子的范围
        scale_factor = np.clip(scale_factor, 0.1, 10.0)
        print(f"  Scale factor clipped to: {scale_factor:.6f}")

    print(f"【Geometry】: Submap scale factor: {scale_factor:.6f}")

    # 计算当前submap的global信息
    # 进行尺寸放缩，对齐到上一个submap尺度
    current_local_poses_scaled = current_local_poses.copy()
    current_local_poses_scaled[:, :3, 3] *= scale_factor
    curr_global_poses = prev_last_global_pose @ current_local_poses_scaled

    return scale_factor, curr_global_poses

# 仅用于初始化计算预积分相对零偏的雅可比函数
def calculate_preintegration_and_jacobian(measurements, start_time, initial_bias_gyro):
    if not measurements:
        return None, None, None, None

    # 【修正1】: 明确初始化所有累加变量的数据类型为 float
    delta_R_mat = np.eye(3, dtype=float)
    delta_V_vec = np.zeros(3, dtype=float)
    delta_P_vec = np.zeros(3, dtype=float)
    J_R_bg = np.zeros((3, 3), dtype=float)
    
    # 确保零偏也是正确的类型
    initial_bias_gyro = np.asarray(initial_bias_gyro, dtype=float)
    
    last_ts = start_time
    for ts, data in measurements:
        dt = ts - last_ts
        if dt <= 0:
            last_ts = ts
            continue

        # 【修正2】: 强制将输入数据转换为 float 类型的 Numpy 数组
        accel = np.asarray(data.accel, dtype=float)
        gyro = np.asarray(data.gyro, dtype=float)

        gyro_corrected = gyro - initial_bias_gyro
        delta_R_step = Rotation.from_rotvec(gyro_corrected * dt).as_matrix()

        # 更新雅可比矩阵
        J_R_bg = delta_R_step.T @ J_R_bg - np.eye(3) * dt

        # 更新预积分增量
        accel_body = delta_R_mat @ accel
        
        # 现在所有的运算都在统一的 float64 类型下进行，不会再有 dtype='O' 的问题
        delta_P_vec += delta_V_vec * dt + 0.5 * accel_body * dt**2
        delta_V_vec += accel_body * dt
        delta_R_mat = delta_R_mat @ delta_R_step
        
        last_ts = ts

    return delta_R_mat, delta_V_vec, delta_P_vec, J_R_bg

def caculate_rotation_matrix_from_two_vectors(vec1, vec2):
    # 计算旋转轴（叉积并归一化）
    axis = np.cross(vec1, vec2)
    # 处理 vec1 和 vec2 平行或反平行的情况
    if np.linalg.norm(axis) < 1e-6:

        # 向量平行（旋转0度）或反平行（旋转180度）
        if np.dot(vec1, vec2) > 0:
            # 平行，旋转0度
            R0 = np.eye(3)
        else:
            # 反平行，旋转180度。选择任意一个与 vec1 垂直的轴
            temp_axis = np.cross(vec1, np.array([1, 0, 0]))
            if np.linalg.norm(temp_axis) < 1e-6:
                temp_axis = np.cross(vec1, np.array([0, 1, 0]))
            axis = temp_axis / np.linalg.norm(temp_axis)
            angle = np.pi
    else:
        axis = axis / np.linalg.norm(axis)

        # 计算旋转角（点积）
        dot_product = np.dot(vec1, vec2)

        # 确保点积在 [-1, 1] 范围内以避免浮点误差
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)

    # 构造旋转向量
    rot_vec = axis * angle
    r = Rotation.from_rotvec(rot_vec)
    R0 = r.as_matrix()
    return R0