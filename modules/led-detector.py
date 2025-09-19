import cv2
import numpy as np
from collections import deque

class LEDDetector:
    def __init__(self):
        # 1. 圆形检测参数
        self.circle_params = {
            'dp': 1.2,
            'minDist': 30,  # 缩小最小距离，适配密集LED
            'param1': 40,   # 降低边缘检测阈值，适配弱光LED
            'param2': 25,   # 降低圆检测阈值，适配小尺寸LED
            'minRadius': 3,
            'maxRadius': 40
        }
        
        # 2. 闪烁检测参数
        self.brightness_history = deque(maxlen=30)
        self.flash_threshold = 20  # 降低阈值，适配低亮度波动
        self.min_flash_frames = 2  # 减少连续帧要求，提升响应速度
        
        # 3. LED 状态追踪
        self.led_states = {}  # key: 稳定ID, value: {pos: (x,y), brightness_history, is_flashing}
        self.next_led_id = 0  # 下一个未分配的LED ID
        self.pos_threshold = 20  # 帧间位置匹配阈值（像素）

    def detect_circles(self, frame):
        """检测圆形物体"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        gray = cv2.medianBlur(gray, 3)  # 新增：滤除椒盐噪声
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            self.circle_params['dp'], self.circle_params['minDist'],
            param1=self.circle_params['param1'], param2=self.circle_params['param2'],
            minRadius=self.circle_params['minRadius'], maxRadius=self.circle_params['maxRadius']
        )
        return np.uint16(np.around(circles)) if circles is not None else None

    def get_led_brightness(self, frame, circle):
        """获取LED亮度"""
        x, y, r = circle
        h, w = frame.shape[:2]
        # 确保ROI完全在图像内
        x_start = max(0, x - r)
        x_end = min(w, x + r)
        y_start = max(0, y - r)
        y_end = min(h, y + r)
        roi = frame[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 2])  # V通道（亮度）均值

    def _match_led_id(self, current_pos):
        """通过位置匹配获取稳定的LED ID"""
        for led_id, state in self.led_states.items():
            # 计算当前位置与历史位置的欧氏距离
            dist = np.linalg.norm(np.array(current_pos) - np.array(state['pos']))
            if dist < self.pos_threshold:
                state['pos'] = current_pos  # 更新最新位置
                return led_id
        # 无匹配：分配新ID
        self.led_states[self.next_led_id] = {
            'pos': current_pos,
            'brightness_history': deque(maxlen=10),
            'is_flashing': False,
            'flash_changes': 0
        }
        new_id = self.next_led_id
        self.next_led_id += 1
        return new_id

    def detect_flashing(self, led_id, brightness):
        """检测LED闪烁"""
        led_state = self.led_states[led_id]
        led_state['brightness_history'].append(brightness)
        
        if len(led_state['brightness_history']) < 3:
            return False
        
        recent_changes = [
            abs(led_state['brightness_history'][i] - led_state['brightness_history'][i-1])
            for i in range(1, len(led_state['brightness_history']))
        ]
        
        if any(change > self.flash_threshold for change in recent_changes):
            led_state['flash_changes'] += 1
            if led_state['flash_changes'] >= self.min_flash_frames:
                led_state['is_flashing'] = True
                led_state['flash_changes'] = 0
                return True
        else:
            led_state['flash_changes'] = max(0, led_state['flash_changes'] - 1)
            led_state['is_flashing'] = False
        return led_state['is_flashing']

    def process_frame(self, frame):
        """处理单帧"""
        circles = self.detect_circles(frame)
        result = frame.copy()
        
        if circles is not None:
            for circle in circles[0, :]:
                x, y, r = circle
                current_pos = (x, y)
                
                # 1. 获取稳定的LED ID
                led_id = self._match_led_id(current_pos)
                
                # 2. 亮度与闪烁检测
                brightness = self.get_led_brightness(frame, (x, y, r))
                is_flashing = self.detect_flashing(led_id, brightness)
                
                # 3. 绘制结果
                color = (0, 0, 255) if is_flashing else (0, 255, 0)
                cv2.circle(result, (x, y), r, color, 2)
                cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(
                    result, f"LED {led_id}: {'Flash' if is_flashing else 'Steady'}",
                    (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                cv2.putText(
                    result, f"Bright: {int(brightness)}",
                    (x - r, y + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        return result