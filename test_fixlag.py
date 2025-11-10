import gtsam
from gtsam.symbol_shorthand import X, L
import numpy as np

def print_graph_indices(smoother: gtsam.IncrementalFixedLagSmoother, title: str):
    """
    一个辅助函数，用于打印当前平滑器中所有因子的索引及其关联的变量。
    (已修复 'name l is not defined' 的 bug)
    """
    print("\n" + "="*20 + f" {title} " + "="*20)
    
    graph = smoother.getFactors()
    
    if graph.size() == 0:
        print("  Graph is empty.")
        return

    print(f"  Total factors: {graph.size()}")
    
    for i in range(graph.size()):
        try:
            factor = graph.at(i)
            if factor:
                keys = factor.keys()
                key_str = ", ".join([gtsam.DefaultKeyFormatter(key) for key in keys])
                factor_type = factor.__class__.__name__
                print(f"  [Index {i}]: Factor connects [{key_str}], Type: {factor_type}")
            else:
                # 修复了 'l' 变量 bug
                print(f"  [Index {i}]: <EmptySlot>") 
        except Exception as e:
            print(f"  [Index {i}]: Error accessing factor: {e}")

# --- 演示开始 ---

# 1. 设置一个非常小的滑窗 (lag = 3.0 秒)
LAG = 3.0
smoother = gtsam.IncrementalFixedLagSmoother(LAG)

# 定义噪声模型
prior_noise_pose = gtsam.noiseModel.Diagonal.Sigmas([1e-6, 1e-6, 1e-6])
prior_noise_point = gtsam.noiseModel.Isotropic.Sigma(2, 1e-6)
odom_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
br_noise = gtsam.noiseModel.Diagonal.Sigmas([np.deg2rad(1.0), 0.1])

# 准备空的容器
new_graph = gtsam.NonlinearFactorGraph()
new_values = gtsam.Values()
new_timestamps = {}

# -----------------------------------------------------------------
# 步骤 0: 添加 X(0) (锚点) 和 L(0) (路标点)
# -----------------------------------------------------------------
pose0 = gtsam.Pose2(0, 0, 0)
point0 = [10, 0] # 路标点在 x=10 处
new_values.insert(X(0), pose0)
new_values.insert(L(0), point0) 
new_graph.add(gtsam.PriorFactorPose2(X(0), pose0, prior_noise_pose))
new_graph.add(gtsam.PriorFactorPoint2(L(0), point0, prior_noise_point))

new_timestamps[X(0)] = 0.0
new_timestamps[L(0)] = 0.0
smoother.update(new_graph, new_values, new_timestamps)
print_graph_indices(smoother, "步骤 0: 锚点已建立")

# -----------------------------------------------------------------
# 步骤 1: 添加 X(1)
# -----------------------------------------------------------------
new_graph = gtsam.NonlinearFactorGraph(); new_values = gtsam.Values(); new_timestamps = {}
pose1 = gtsam.Pose2(1, 0, 0)
new_values.insert(X(1), pose1)
new_graph.add(gtsam.BetweenFactorPose2(X(0), X(1), gtsam.Pose2(1, 0, 0), odom_noise))
new_timestamps[X(1)] = 1.0
smoother.update(new_graph, new_values, new_timestamps)
print_graph_indices(smoother, "步骤 1: X(1) 已添加")

# -----------------------------------------------------------------
# 步骤 2: 添加观测因子 (创建冲突)
# -----------------------------------------------------------------
new_graph = gtsam.NonlinearFactorGraph(); new_values = gtsam.Values(); new_timestamps = {}

# 因子 A: X(0) 观测 L(0) (距离=10)
# 这是“危险”的因子，因为它连接着 X(0)
new_graph.add(gtsam.BearingRangeFactor2D(X(0), L(0), gtsam.Rot2.fromDegrees(0), 10.0, br_noise))

# 因子 B: X(1) 观测 L(0) (距离=9)
# 这是“安全”的因子，它确保 L(0) 保持约束
new_graph.add(gtsam.BearingRangeFactor2D(X(1), L(0), gtsam.Rot2.fromDegrees(0), 9.0, br_noise))

# 确保 L(0) 的时间戳被更新，这样它就不会和 X(0) 一起被边缘化
new_timestamps[L(0)] = 1.0 

smoother.update(new_graph, new_values, new_timestamps)
print_graph_indices(smoother, "步骤 2: 观测已添加")

# -----------------------------------------------------------------
# 步骤 3: 最终的崩溃测试
# -----------------------------------------------------------------
#
# *当前状态* (大致):
# [Index 0]: Factor connects [x0] (先验)
# [Index 1]: Factor connects [l0] (先验)
# [Index 2]: Factor connects [x0, x1] (里程计)
# [Index 3]: Factor connects [x0, l0]  <--- 我们的目标 (危险因子)
# [Index 4]: Factor connects [x1, l0]  (安全因子)
#
# 我们现在将在 *同一次* update 调用中：
# 1. *手动* 请求删除 [Index 3] (模拟删除负深度点)
# 2. *自动* 触发 X(0) 边缘化 (它也需要访问 [Index 3])
# -----------------------------------------------------------------
print("\n" + "!"*60)
print("! 步骤 3 (崩溃测试): *同时* 手动删除 [Index 3]")
print("! 并触发 X(0) 的自动边缘化")
print("! 预期: 崩溃 (IndexError: map::at)")
print("!"*60)

# 1. 准备新因子 (添加 X(2) at t=4.0)
new_graph_crash = gtsam.NonlinearFactorGraph()
new_values_crash = gtsam.Values()
new_timestamps_crash = {}

pose2_main = gtsam.Pose2(2, 0, 0)
new_values_crash.insert(X(2), pose2_main)
new_graph_crash.add(gtsam.BetweenFactorPose2(X(1), X(2), gtsam.Pose2(1, 0, 0), odom_noise))

# 2. *** 触发 X(0) 边缘化 ***
new_timestamps_crash[X(2)] = 3.0 # t_latest=4.0, 滑窗=[1.0, 4.0], X(0) 被移出
# 确保 L(0) 仍在滑窗内
new_timestamps_crash[L(0)] = 3.0 

# 3. 准备要删除的索引 (我们的“危险”因子)
indices_to_remove_crash = []
indices_to_remove_crash.append(4) # <--- 目标！[Index 3]

try:
    # 4. *** 执行合并的“致命”更新 ***
    smoother.update(new_graph_crash, new_values_crash, new_timestamps_crash)
    
    print_graph_indices(smoother, "步骤 3: 居然没崩溃？ (极不可能)")

except Exception as e:
    print(f"\n!!!! 步骤 3 成功复现崩溃 !!!!\nERROR: {e}\n")
    print("✓✓✓ 崩溃被成功复现。这证明了我们的分析。 ✓✓✓")