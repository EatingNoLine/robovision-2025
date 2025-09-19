import cv2
import numpy as np
from apriltag-detector import AprilTagDetector
from led-detector import LEDDetector

class MainController:
    def __init__(self, config):
        """
        主控制器初始化
        
        Args:
            config (dict): 配置参数
                - camera_index: 摄像头索引（0为默认）或视频路径
                - calib_file: 相机标定文件路径
                - tag_size: AprilTag实际边长（米）
                - tag_family: AprilTag家族（如"tag36h11"）
                - display_mode: 显示模式（"both"/"apriltag"/"led"）
        """
        self.config = config
        self._init_camera()
        self._init_detectors()
        self.running = False

    def _init_camera(self):
        """初始化摄像头/视频流"""
        cam_source = self.config.get("camera_index", 0)
        self.cap = cv2.VideoCapture(cam_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头/视频：{cam_source}")
        # 设置摄像头分辨率（可选）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def _init_detectors(self):
        """初始化AprilTag和LED检测器"""
        # 1. AprilTag检测器（加载标定参数）
        self.april_tag_detector = AprilTagDetector.from_calibration_file(
            calib_file=self.config["calib_file"],
            tag_family=self.config["tag_family"],
            tag_size=self.config["tag_size"]
        )
        # 2. LED检测器
        self.led_detector = LEDDetector()

    def _process_single_frame(self, frame):
        """处理单帧：并行执行两种检测，融合结果"""
        display_mode = self.config.get("display_mode", "both")
        result_frame = frame.copy()

        # 1. 执行检测（顺序可互换，无依赖）
        if display_mode in ["both", "apriltag"]:
            april_tag_results = self.april_tag_detector.detect_tags(frame)
        if display_mode in ["both", "led"]:
            led_result_frame = self.led_detector.process_frame(frame)

        # 2. 融合绘制结果
        if display_mode == "both":
            # 以LED检测结果为底，叠加AprilTag结果
            result_frame = self.april_tag_detector.draw_detections(led_result_frame, april_tag_results)
        elif display_mode == "apriltag":
            result_frame = self.april_tag_detector.draw_detections(result_frame, april_tag_results)
        elif display_mode == "led":
            result_frame = led_result_frame

        # 3. 叠加系统信息（帧率、检测模式）
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            result_frame, f"FPS: {int(fps)} | Mode: {display_mode}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )
        return result_frame, april_tag_results if "april_tag_results" in locals() else None

    def start(self):
        """启动主流程"""
        print("主控制器启动 | 按 'q' 退出 | 按 'm' 切换显示模式")
        self.running = True
        current_mode = self.config.get("display_mode", "both")
        mode_list = ["both", "apriltag", "led"]

        while self.running:
            # 1. 读取帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取帧，退出程序")
                break

            # 2. 处理帧
            result_frame, april_tag_results = self._process_single_frame(frame)

            # 3. 显示结果
            cv2.imshow("AprilTag + LED Detector", cv2.resize(result_frame, (1280, 720)))

            # 4. 键盘交互
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('m'):
                # 切换显示模式
                current_idx = mode_list.index(current_mode)
                current_mode = mode_list[(current_idx + 1) % 3]
                self.config["display_mode"] = current_mode
                print(f"切换显示模式：{current_mode}")

        # 5. 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("主控制器已退出")

    def get_latest_results(self):
        """获取最新检测结果（用于后续扩展，如控制逻辑）"""
        # 可扩展：返回AprilTag位姿、LED闪烁状态等
        return {
            "april_tag": self.april_tag_detector.detections if hasattr(self.april_tag_detector, "detections") else None,
            "led_states": self.led_detector.led_states
        }

# 配置参数（根据实际场景修改）
CONFIG = {
    "camera_index": 0,  # 0=默认摄像头，或视频文件路径如"test.mp4"
    "calib_file": "camera_calibration.npz",  # 相机标定文件
    "tag_size": 0.1,  # AprilTag实际边长（米）
    "tag_family": "tag36h11",
    "display_mode": "both"  # 初始显示模式
}

if __name__ == "__main__":
    try:
        controller = MainController(CONFIG)
        controller.start()
    except Exception as e:
        print(f"程序异常：{e}")