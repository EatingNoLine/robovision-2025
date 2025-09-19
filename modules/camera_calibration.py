import cv2
import numpy as np
import os
import glob

class CameraCalibrator:
    """相机标定类，用于计算相机内参和畸变系数"""
    
    def __init__(self, chessboard_size=(9, 6), square_size=0.025):
        """
        初始化标定器
        
        Args:
            chessboard_size: 棋盘格内角点数量 (宽度方向角点数, 高度方向角点数)
            square_size: 棋盘格每个方格的实际尺寸（单位：米）
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 存储所有图像的三维世界坐标和二维图像坐标
        self.object_points = []  # 3D点在真实世界中的坐标
        self.image_points = []   # 2D点在图像中的坐标
        
        # 生成棋盘格的三维坐标（世界坐标系）
        # 假设棋盘格在z=0平面上
        self.objectp = np.zeros((np.prod(chessboard_size), 3), np.float32)
        self.objectp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objectp *= square_size  # 转换为实际尺寸（米）
        
        # 标定结果
        self.camera_matrix = None    # 内参矩阵
        self.dist_coeffs = None      # 畸变系数
        self.rvecs = None            # 旋转向量
        self.tvecs = None            # 平移向量

    def collect_corners(self, image_paths):
        """
        从图像中检测棋盘格角点并收集数据
        
        Args:
            image_paths: 包含棋盘格图像的路径列表
            
        Returns:
            成功检测到角点的图像数量
        """
        success_count = 0
        
        for img_path in image_paths:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue
                
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 检测棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            # 如果检测到角点，进行亚像素优化
            if ret:
                success_count += 1
                self.object_points.append(self.objectp)
                
                # 亚像素角点检测，提高精度
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.image_points.append(corners2)
                
                # 绘制角点并显示
                img = cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                cv2.imshow('Chessboard Corners', cv2.resize(img, (800, 600)))
                cv2.waitKey(500)  # 显示500ms
            else:
                print("未检测到角点: ", img_path)
                
        cv2.destroyAllWindows()
        return success_count

    def calibrate(self, image_size):
        """
        执行相机标定
        
        Args:
            image_size: 图像尺寸 (宽度, 高度)
            
        Returns:
            标定的重投影误差（越小越好，通常应小于1像素）
        """
        if len(self.object_points) < 5:
            raise ValueError("至少需要5张成功检测到角点的图像进行标定")
            
        # 执行标定
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, image_size, None, None
        )
        
        # 计算重投影误差（评估标定精度）
        mean_error = 0
        for i in range(len(self.object_points)):
            img_points2, _ = cv2.projectPoints(
                self.object_points[i], self.rvecs[i], self.tvecs[i], 
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error
            
        reproj_error = mean_error / len(self.object_points)
        return ret, reproj_error

    def save_calibration(self, save_path):
        """保存标定结果到文件"""
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("请先完成标定再保存结果")
            
        # 保存为npz格式
        np.savez(
            save_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            rvecs=self.rvecs,
            tvecs=self.tvecs
        )
        print(f"标定结果已保存到: {save_path}")

    def load_calibration(self, load_path):
        """从文件加载标定结果"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"标定文件不存在: {load_path}")
            
        data = np.load(load_path)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        self.rvecs = data['rvecs']
        self.tvecs = data['tvecs']
        print(f"已加载标定结果: {load_path}")

    def undistort_image(self, image):
        """
        对图像进行畸变矫正
        
        Args:
            image: 输入图像
            
        Returns:
            矫正后的图像
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("请先加载或完成标定")
            
        h, w = image.shape[:2]
        # 获取优化后的新内参矩阵（可选）
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 矫正图像
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )
        
        # 根据ROI裁剪图像（去除黑边）
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted


if __name__ == "__main__":
    # --------------------------
    # 配置参数
    # --------------------------
    chessboard_size = (7, 7)    # 棋盘格内角点数量（宽x高）
    square_size = 0.010         # 每个方格的实际尺寸（米）
    image_dir = "calibration_images"  # 存放标定图像的文件夹
    calib_result_path = "camera_calibration.npz"  # 标定结果保存路径
    
    # --------------------------
    # 步骤1: 准备标定图像
    # --------------------------
    # 确保标定图像文件夹存在
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"已创建标定图像文件夹: {image_dir}")
        print("请在该文件夹中放入至少10张不同角度的棋盘格图像，然后重新运行程序")
        exit()
    
    # 获取所有图像路径
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                 glob.glob(os.path.join(image_dir, "*.png"))
    
    if len(image_paths) == 0:
        print(f"在 {image_dir} 中未找到图像文件")
        print("请放入标定图像后重新运行")
        exit()
    
    print(f"找到 {len(image_paths)} 张图像，开始检测角点...")
    
    # --------------------------
    # 步骤2: 检测角点
    # --------------------------
    calibrator = CameraCalibrator(chessboard_size, square_size)
    success_count = calibrator.collect_corners(image_paths)
    print(f"成功检测到 {success_count} 张图像的角点")
    
    if success_count < 5:
        print("成功检测到的图像数量不足，请添加更多图像或检查图像质量")
        exit()
    
    # --------------------------
    # 步骤3: 执行标定
    # --------------------------
    # 获取图像尺寸（使用第一张成功检测的图像）
    sample_img = cv2.imread(image_paths[0])
    image_size = (sample_img.shape[1], sample_img.shape[0])  # (宽度, 高度)
    
    # 执行标定
    ret, reproj_error = calibrator.calibrate(image_size)
    
    if not ret:
        print("标定失败")
        exit()
    
    print(f"标定成功！重投影误差: {reproj_error:.6f} 像素")
    print("相机内参矩阵:")
    print(calibrator.camera_matrix)
    print("\n畸变系数:")
    print(calibrator.dist_coeffs)
    
    # --------------------------
    # 步骤4: 保存标定结果
    # --------------------------
    calibrator.save_calibration(calib_result_path)
    
    # --------------------------
    # 步骤5: 演示畸变矫正效果
    # --------------------------
    # 读取一张测试图像
    test_img = cv2.imread(image_paths[0])
    # 矫正图像
    undistorted_img = calibrator.undistort_image(test_img)
    
    # 显示原图和矫正后的图像
    combined = np.hstack((
        cv2.resize(test_img, (640, 480)),
        cv2.resize(undistorted_img, (640, 480))
    ))
    
    cv2.putText(combined, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Undistorted", (650, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Original vs Undistorted", combined)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
