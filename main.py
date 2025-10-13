import os
import argparse
import yaml
from pathlib import Path
import queue
import time
import multiprocessing as mp

import torch
import cv2

from utils.dataloader import UnifiedDataloader
from datatype.global_map import GlobalMap
from core.estimator import Estimator
from core.feature_tracker import FeatureTracker

def main():
    # 1. 配置参数解析
    parser = argparse.ArgumentParser(description="PyVINS-Fusion: Visual-Inertial SLAM System For Python")

    # 配置文件路径
    parser.add_argument('--config', type=Path, default='config/config.yaml',
                        help="Path to the configuration file")
    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            print("Configuration loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return

    # 2. 初始化模块和通信管道
    print("Initializing SLAM components and communication queues...")
    
    # 启动一个spawn主进程
    mp.set_start_method('spawn', force=True)

    # 创建全局地图
    global_central_map = GlobalMap()

    # 创建队列
    feature_tracker_to_estimator_queue = queue.Queue(maxsize=5) # 前端传递给系统VGGT推理数据和IMU预积分数据，线程之间传递

    # 初始化数据加载器
    dataloader_config = {'path': config['dataset_path'], 'dataset_type': config['dataset_type']}
    data_loader = UnifiedDataloader(dataloader_config)

    # 实例化所有模块
    # viewer = Viewer(system_to_viewer_queue)
    feature_tracker = FeatureTracker(config, data_loader, feature_tracker_to_estimator_queue)
    # estimator = IVGGTSystem(config, frontend_to_system_queue, system_to_viewer_queue,
    #                     global_central_map, imu_processor, backend)
    
    # 建立连接
    # system.set_viewer(viewer)
    
    # 3. 启动所有线程
    print("Starting all SLAM threads...")
    feature_tracker.start()
    # estimator.start()
    # viewer.start()

    # 4. 主线程等待所有子线程结束
    try:
        # 使用一个无限循环让主线程保持存活
        while True:
            # 关闭可视化窗口则退出进程
            if not feature_tracker.thread.is_alive():
                print("[Main Process] All processing threads have finished. Initiating shutdown.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main Process] Caught KeyboardInterrupt, initiating shutdown...")
    finally:
        # 5. 安全关闭所有模块
        print("[Main Process] Shutting down all components...")
        
        # a. 首先，向所有子任务发送停止信号
        feature_tracker.shutdown()
        # estimator.shutdown()
        # viewer.shutdown()

        # b. 然后，等待所有子任务真正执行完毕并退出
        # 等待线程
        feature_tracker.thread.join(timeout=2)
        # estimator.thread.join(timeout=2)
        # 等待进程
        # viewer.join(timeout=2)

        cv2.destroyAllWindows()
        print("[Main Process] SLAM system shut down.")

if __name__ == "__main__":
    main()