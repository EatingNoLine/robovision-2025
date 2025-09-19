import cv2
import apriltag
import numpy as np


class AprilTagDetector:
    """
    集成相机标定参数的AprilTag识别与位姿分析类
    功能：检测图像中的AprilTag，使用标定参数计算更精确的位姿
    """

    def __init__(self, tag_family="tag36h11", tag_size=0.1, 
                 camera_matrix=None, dist_coeffs=None):
        """
        初始化检测器，使用相机标定参数
        
        Args:
            tag_family (str): AprilTag标签家族
            tag_size (float): 实际标签边长（米）
            camera_matrix (np.ndarray): 相机内参矩阵 (3x3)
            dist_coeffs (np.ndarray): 畸变系数 (1x5)
        """
        # 初始化AprilTag检测器
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families=tag_family))
        
        # 核心参数配置
        self.tag_family = tag_family
        self.tag_size = tag_size
        self.camera_matrix = camera_matrix  # 相机内参矩阵
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1))  # 畸变系数
        
        # 提取内参参数
        if self.camera_matrix is not None:
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
        else:
            self.fx = self.fy = self.cx = self.cy = None

        # 标签四个角点的3D坐标（相对于标签中心）
        half_size = self.tag_size / 2.0
        self.tag_3d_points = np.array([
            [-half_size, -half_size, 0],  # 左上角
            [half_size, -half_size, 0],   # 右上角
            [half_size, half_size, 0],    # 右下角
            [-half_size, half_size, 0]    # 左下角
        ], dtype=np.float32)

    @classmethod
    def from_calibration_file(cls, calib_file, tag_family="tag36h11", tag_size=0.1):
        """从标定文件创建检测器实例"""
        calib_data = np.load(calib_file)
        camera_matrix = calib_data["camera_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]
        return cls(tag_family, tag_size, camera_matrix, dist_coeffs)

    def detect_tags(self, image):
        """检测图像中的AprilTag并计算位姿"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 执行标签检测
        detections = self.detector.detect(gray)

        # 解析检测结果并计算位姿
        results = []
        for det in detections:
            # 提取标签核心信息
            tag_id = int(det.tag_id)
            center = tuple(det.center.astype(int))
            corners = det.corners.astype(int)
            corners = [tuple(pt) for pt in corners]

            # 位姿解算（仅当提供相机内参时）
            pose = None
            if self.camera_matrix is not None:
                # 求解PnP问题，使用实际畸变系数
                ret, rvec, tvec = cv2.solvePnP(
                    objectPoints=self.tag_3d_points,
                    imagePoints=det.corners.astype(np.float32),
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=self.dist_coeffs,  # 使用标定得到的畸变系数
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if ret:  # 位姿解算成功
                    pose = {
                        "rvec": rvec,  # 旋转向量（3x1，弧度）
                        "tvec": tvec   # 平移向量（3x1，米）
                    }

            # 存储单标签结果
            results.append({
                "id": tag_id,
                "center": center,
                "corners": corners,
                "pose": pose
            })

        return results

    def draw_detections(self, image, detections):
        """在图像上绘制检测结果"""
        output_img = image.copy()
        for det in detections:
            tag_id = det["id"]
            center = det["center"]
            corners = det["corners"]
            pose = det["pose"]

            # 1. 绘制标签框
            for i in range(4):
                start_pt = corners[i]
                end_pt = corners[(i + 1) % 4]
                cv2.line(output_img, start_pt, end_pt, color=(0, 255, 0), thickness=2)

            # 2. 绘制标签中心
            cv2.circle(output_img, center, radius=4, color=(0, 0, 255), thickness=-1)

            # 3. 绘制标签ID
            cv2.putText(
                output_img,
                f"ID: {tag_id}",
                (center[0] - 20, center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            # 4. 绘制位姿轴（使用畸变系数）
            if pose is not None and self.camera_matrix is not None:
                # 绘制坐标轴：x轴（红）、y轴（绿）、z轴（蓝）
                axis_length = self.tag_size / 2.0
                cv2.drawFrameAxes(
                    output_img,
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=self.dist_coeffs,  # 使用实际畸变系数
                    rvec=pose["rvec"],
                    tvec=pose["tvec"],
                    length=axis_length,
                    thickness=2
                )

        return output_img

    def analyze_pose(self, pose):
        """解析位姿数据，转换为更易理解的形式"""
        if pose is None:
            return None

        rvec = pose["rvec"]
        tvec = pose["tvec"]

        # 旋转向量 -> 旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # 旋转矩阵 -> 欧拉角（Z-Y-X顺规，对应yaw-pitch-roll）
        sy = np.sqrt(rotation_matrix[0,0] **2 + rotation_matrix[1,0]** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            pitch = np.arctan2(-rotation_matrix[2,0], sy)
            yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            roll = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            pitch = np.arctan2(-rotation_matrix[2,0], sy)
            yaw = 0.0

        # 计算直线距离
        distance = np.linalg.norm(tvec)

        return {
            "rotation_matrix": rotation_matrix,
            "euler_angles": (roll, pitch, yaw),
            "distance": distance,
            "euler_angles_deg": np.degrees((roll, pitch, yaw)),  # 角度制
            "translation": tvec.flatten()  # 平移向量
        }


# 使用示例
if __name__ == "__main__":
    # 配置参数
    TAG_SIZE = 0.1  # 标签实际边长（米），根据你的标签修改
    CALIB_FILE = "camera_calibration.npz"  # 标定结果文件
    TEST_IMAGE = "test_image.jpg"  # 测试图像路径

    # 从标定文件创建检测器
    detector = AprilTagDetector.from_calibration_file(
        calib_file=CALIB_FILE,
        tag_family="tag36h11",
        tag_size=TAG_SIZE
    )

    # 读取测试图像
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print(f"无法读取图像: {TEST_IMAGE}")
        exit()

    # 检测标签
    detections = detector.detect_tags(image)
    print(f"检测到 {len(detections)} 个标签")

    # 分析并打印结果
    for det in detections:
        print(f"\n标签 ID: {det['id']}")
        print(f"中心坐标: {det['center']}")
        
        if det["pose"] is not None:
            pose_info = detector.analyze_pose(det["pose"])
            print(f"距离相机: {pose_info['distance']:.3f} 米")
            print(f"欧拉角 (roll, pitch, yaw): {pose_info['euler_angles_deg']:.2f} 度")
            print(f"平移向量 (x, y, z): {pose_info['translation']:.3f} 米")

    # 绘制结果并显示
    result_img = detector.draw_detections(image, detections)
    cv2.imshow("AprilTag Detection", cv2.resize(result_img, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
