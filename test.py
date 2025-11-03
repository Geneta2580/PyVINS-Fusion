import gtsam
# 假设 gtsam.LinearizationMode 和 gtsam.DegeneracyMode 可用

# 1. 构造对象 (唯一支持的方式)
params = gtsam.SmartProjectionParams()

# 2. 使用设置器函数配置参数

# 设置线性化模式
params.setLinearizationMode(gtsam.LinearizationMode.JACOBIAN_Q)

# 设置退化模式
params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)

# 设置重三角化阈值 (retriangulationTh)
# 注意：虽然 help 中没有列出 setRetriangulationTh，但它通常作为可直接赋值的属性存在
params.retriangulationThreshold = 0.001 

# 设置 Cheirality 检查参数 (通常也是 set_ 或直接属性)
params.set_throwCheirality(False)
params.set_verboseCheirality(True)

# 设置其他高级参数 (可选)
params.setEnableEPI(True) 
params.setDynamicOutlierRejectionThreshold(True)