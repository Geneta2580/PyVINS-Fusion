import numpy as np
from scipy.spatial.transform import Rotation

def pose_matrix_to_tum_format(pose_matrix):
    """Converts a 4x4 SE(3) pose matrix to a TUM trajectory format string components."""
    t = pose_matrix[:3, 3]
    q = Rotation.from_matrix(pose_matrix[:3, :3]).as_quat() # q is [x, y, z, w]
    return t[0], t[1], t[2], q[0], q[1], q[2], q[3]



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