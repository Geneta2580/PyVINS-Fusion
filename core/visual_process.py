import cv2
import numpy as np
import time
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

class VisualProcessor:
    def __init__(self, checkpoint_path, config):
        self.config = config

        self.prev_kf_pts = None
        self.prev_gray = None
        self.keyframe_buffer_size = 3
        self.keyframe_buffer = []
        
        # 加载VGGT模型
        print("Visual Processor is loading the model...")
        gpu_id = config.get('gpu_id', 0)
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = f"cuda:{gpu_id}"
            print(f"Using specified GPU: {self.device}")
        else:
            self.device = "cpu"
            print("Specified GPU not available, falling back to CPU.")
            
        state_dict = torch.load(checkpoint_path)
        self.model = VGGT().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Model loaded successfully in Visual Processor.")

    # 提取特征点
    def detect_features(self, gray_image, max_corners=1000, mask=None):
        return cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
            mask=mask
        )

    # 重新提取特征点
    def initialize_keyframe(self, image):
        self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.prev_kf_pts = self.detect_features(self.prev_gray)

    # 使用光流计算视差
    def compute_parallax(self, image, min_parallax, visualize=False):
        if self.prev_gray is None or self.prev_kf_pts is None or len(self.prev_kf_pts) < 10:
            self.initialize_keyframe(image)
            return True

        # 最新的kf转为灰度
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Track keyframe points into current frame，上一帧的pts作为初值
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_kf_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        status = status.flatten()
        good_prev = self.prev_kf_pts[status == 1]
        good_curr = curr_pts[status == 1]

        # 如果跟踪到的特征点太少，则暴力重置，标记Keyframe
        if len(good_prev) < 10:
            self.initialize_keyframe(image)
            return True

        # Measure displacement to mean parallax
        displacement = np.linalg.norm(good_curr - good_prev, axis=1)
        mean_parallax = np.mean(displacement) if len(displacement) > 0 else 0

        if visualize:
            vis = image.copy()
            for p1, p2 in zip(good_prev, good_curr):
                p1 = tuple(p1.ravel().astype(int))
                p2 = tuple(p2.ravel().astype(int))
                cv2.arrowedLine(vis, p1, p2, color=(0, 255, 0), thickness=1, tipLength=0.3)
            cv2.imshow("Optical Flow", vis)
            cv2.waitKey(1)

        # 若平均视差大于阈值，则暴力重置，标记Keyframe
        if mean_parallax > min_parallax:
            self.initialize_keyframe(image)
            return True
        else:
            return False
    
    def is_keyframe(self, image_data, timestamp, min_parallax=10, visualize=False) -> bool:
        return self.compute_parallax(
            image=image_data,
            min_parallax=min_parallax, 
            visualize=visualize
        )

    def run_predictions(self, keyframe_buffer_data, max_loops):
        # 分离图像路径和时间戳
        image_names = [item[0] for item in keyframe_buffer_data]
        timestamps = [item[1] for item in keyframe_buffer_data]

        # 对图片进行预处理
        images = load_and_preprocess_images(image_names).to(self.device)
        print(f"Preprocessed images shape: {images.shape}")

        print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # --- 计时开始 ---
        if self.device.startswith("cuda"):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()
        # --- 计时开始 ---

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images)

        # --- 计时结束 ---
        if self.device.startswith("cuda"):
            end_event.record()
            torch.cuda.synchronize()  # 等待CUDA核心完成所有任务
            inference_time_ms = start_event.elapsed_time(end_event)
            print(f"--- Inference GPU time: {inference_time_ms:.2f} ms ---")
        else:
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            print(f"--- Inference CPU time: {inference_time_ms:.2f} ms ---")
        # --- 计时结束 ---

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # 把预测结果从GPU转移到CPU，同时解包
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                
                # 检查相机外参
                # if key == "extrinsic":
                #     # 提取将要被转换的张量
                #     pose_tensor = predictions[key]
                    
                #     # 转换为 NumPy 数组以便打印
                #     poses_np = pose_tensor.cpu().numpy().squeeze(0)
                    
                #     print(f"--- Key: '{key}' (Pose Data) ---")
                #     # 使用 numpy 的打印选项来格式化输出
                #     np.set_printoptions(precision=4, suppress=True)
                #     print(poses_np)

                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        # 在返回 predictions 之前，把时间戳加进去
        predictions["timestamps"] = np.array(timestamps)
        return predictions

    def process_new_keyframe(self, image_path, timestamp):
        print(f"New keyframe. Adding '{image_path}' to buffer.")
        self.keyframe_buffer.append((image_path, timestamp))

        predictions = None
        if len(self.keyframe_buffer) == self.keyframe_buffer_size:
            print("Buffer full. Running model prediction.")
            
            # t_start = time.time()

            max_loops = self.config.get('max_loops', 10)
            predictions = self.run_predictions(self.keyframe_buffer, max_loops)
            
            # t_end = time.time()
            # print(f"[PROFILE] Frontend prediction total time: {t_end - t_start:.4f} seconds")
            
            # --- 滑动窗口 ---
            self.keyframe_buffer = self.keyframe_buffer[-1:]
            print("--- Keyframe buffer slided. ---")

        return predictions