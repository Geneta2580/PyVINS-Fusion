import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Debugger:
    @staticmethod
    def save_full_matrix_python(matrix):
        try:
            filename = "hessian.txt"
            # 使用 np.savetxt 直接保存
            # delimiter=',' 指定了使用逗号作为分隔符
            # fmt='%.8g' 是一个通用的数字格式，可以保留足够的精度，同时保持格式整洁
            np.savetxt(filename, matrix, delimiter=',', fmt='%.8g')
            
            print(f"Python: 矩阵已成功以 txt 格式写入到 '{filename}'")

        except Exception as e:
            print(f"Python: 写入文件时出错: {e}")

    @staticmethod
    def visualize_matrix(matrix, title="Matrix Heatmap", save_path=None):
        try:
            # 创建一个图形窗口
            plt.figure(figsize=(12, 10))
            
            # 使用 seaborn 的 heatmap 函数
            # cmap='viridis' 是一个视觉上很舒服的色谱
            # annot=False 因为矩阵太大，不适合显示数字
            # square=True 保证每个单元格是正方形
            sns.heatmap(matrix, cmap='viridis', annot=False, square=True)
            
            plt.title(title, fontsize=16)
            plt.xlabel("State Variables", fontsize=12)
            plt.ylabel("State Variables", fontsize=12)
            
            if save_path:
                plt.savefig(save_path)
                print(f"【Debug】: Heatmap saved to {save_path}")
            
            # 显示图形
            # plt.show()

        except Exception as e:
            print(f"【Debug】: Failed to visualize matrix. Error: {e}")
