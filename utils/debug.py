import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import csv

class Debugger:
    def __init__(self, log_dir="output", file_prefix="log", column_names=None, use_timestamp=True):
        """
        初始化日志记录器。

        参数:
            log_dir (str): 存储日志文件的目录。如果不存在，会自动创建。
            file_prefix (str): 日志文件名的前缀。
            column_names (list of str): 用户提供的数据列的表头列表。例如: ['error', 'velocity']。
            use_timestamp (bool): 是否在文件名中添加时间戳（推荐）。
        """
        if column_names is None:
            column_names = ['value'] # 如果未提供列名，则默认为'value'

        # --- 1. 创建日志目录 ---
        os.makedirs(log_dir, exist_ok=True)

        # --- 2. 构建文件名 ---
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{file_prefix}_{timestamp}.csv"
        else:
            file_name = f"{file_prefix}.csv"
        
        self.log_path = os.path.join(log_dir, file_name)
        self.round_counter = 0

        # --- 3. 打开文件并写入表头 ---
        # 使用'w'模式（写入）和 newline='' 来防止写入空行
        self.file_handle = open(self.log_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file_handle)
        
        # 写入完整的表头，第一列总是'Round'
        header = ['Round'] + column_names
        self.writer.writerow(header)
        
        print(f"【Logger】Logging to {self.log_path}")

    def log(self, *values):
        row = [self.round_counter] + list(values)
        self.writer.writerow(row)
        self.round_counter += 1

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            print(f"【Logger】Closed log file {self.log_path}")
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
