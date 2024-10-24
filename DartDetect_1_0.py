import cv2
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Trajectory:
    """
    表示飞镖轨迹的数据类。

    Attributes:
        points (List[Tuple[float, float]]): 轨迹点的列表，每个点是一个 (x, y) 坐标元组。
        lifetime (int): 轨迹的剩余生命值。
    """
    points: List[Tuple[float, float]]
    lifetime: int

class DartDetector:
    """
    飞镖检测器类，用于从视频中检测和跟踪飞镖轨迹。

    Attributes:
        cap (cv2.VideoCapture): 视频捕获对象。
        fps (int): 视频的帧率。
        width (int): 视频帧的宽度。
        height (int): 视频帧的高度。
        writer (cv2.VideoWriter): 视频写入对象，用于保存处理后的视频。
        config (dict): 包含各种可调参数的配置字典。
        trajectories (List[Trajectory]): 当前跟踪的轨迹列表。
        last_light_level (float): 上一帧的平均亮度，用于检测光照变化。
    """

    def __init__(self, video_file: str, output_file: str):
        """
        初始化 DartDetector 对象。

        Args:
            video_file (str): 输入视频文件的路径。
            output_file (str): 输出视频文件的路径。
        """
        self.cap = cv2.VideoCapture(video_file)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))
        
        self.config = {
            'max_dist': 150,
            'fit_pts_num': 100,
            'traj_error_limit': 3,
            'light_change_threshold': 5,
            'edge_threshold': 10,
            'min_traj_length': 10,
            'max_traj_lifetime': 10
        }

        self.trajectories: List[Trajectory] = []
        self.last_light_level = None

    @staticmethod
    def quadratic(x: float, a: float, b: float, c: float) -> float:
        """
        二次函数。

        Args:
            x (float): 自变量。
            a (float): 二次项系数。
            b (float): 一次项系数。
            c (float): 常数项。

        Returns:
            float: 函数值 f(x) = ax^2 + bx + c。
        """
        return a * x**2 + b * x + c

    def fit_points(self, points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        """
        使用最小二乘法拟合点集到二次函数。

        Args:
            points (List[Tuple[float, float]]): 要拟合的点集。

        Returns:
            Tuple[float, float, float]: 拟合得到的二次函数系数 (a, b, c)。
        """
        x, y = zip(*points)
        return curve_fit(self.quadratic, x, y)[0]

    def draw_trajectory(self, img: np.ndarray, traj: List[Tuple[float, float]], color: Tuple[int, int, int], alpha: float = 1):
        """
        在图像上绘制轨迹。

        Args:
            img (np.ndarray): 要绘制的图像。
            traj (List[Tuple[float, float]]): 轨迹点列表。
            color (Tuple[int, int, int]): 绘制颜色，BGR 格式。
            alpha (float, optional): 轨迹的透明度。默认为 1（不透明）。
        """
        overlay = img.copy()
        for i in range(len(traj) - 1):
            cv2.circle(overlay, (int(traj[i][0]), int(traj[i][1])), 2, color, -1)
            cv2.line(overlay, (int(traj[i][0]), int(traj[i][1])), 
                     (int(traj[i+1][0]), int(traj[i+1][1])), color, 1)
        cv2.circle(overlay, (int(traj[-1][0]), int(traj[-1][1])), 2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_curve(self, img: np.ndarray, coeffs: Tuple[float, float, float], color: Tuple[int, int, int], start_x: int, direction: int, alpha: float = 1):
        """
        在图像上绘制二次曲线。

        Args:
            img (np.ndarray): 要绘制的图像。
            coeffs (Tuple[float, float, float]): 二次函数的系数 (a, b, c)。
            color (Tuple[int, int, int]): 绘制颜色，BGR 格式。
            start_x (int): 绘制起点的 x 坐标。
            direction (int): 绘制方向，1 表示向右，-1 表示向左。
            alpha (float, optional): 曲线的透明度。默认为 1（不透明）。
        """
        overlay = img.copy()
        a, b, c = coeffs
        for x in range(start_x, 0 if direction < 0 else img.shape[1], direction):
            y = int(self.quadratic(x, a, b, c))
            if 0 <= y < img.shape[0]:
                cv2.circle(overlay, (x, y), 1, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def is_point_out_of_frame(self, point: Tuple[float, float]) -> bool:
        """
        检查点是否超出图像边界。

        Args:
            point (Tuple[float, float]): 要检查的点的坐标。

        Returns:
            bool: 如果点超出图像边界则返回 True，否则返回 False。
        """
        x, y = point
        return (x < self.config['edge_threshold'] or 
                x > self.width - self.config['edge_threshold'] or 
                y < self.config['edge_threshold'] or 
                y > self.height - self.config['edge_threshold'])
    
    def detect_light_change(self, frame: np.ndarray) -> bool:
        """
        检测图像的光照变化。

        Args:
            frame (np.ndarray): 当前帧。

        Returns:
            bool: 如果检测到显著的光照变化则返回 True，否则返回 False。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_light_level = np.mean(gray)
        
        if self.last_light_level is None:
            self.last_light_level = current_light_level
            return False
        
        light_change = abs(current_light_level - self.last_light_level)
        self.last_light_level = current_light_level
        
        return light_change > self.config['light_change_threshold']

    def process_frame(self, frame: np.ndarray, prev_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        处理单个视频帧。

        Args:
            frame (np.ndarray): 当前帧。
            prev_frame (np.ndarray): 上一帧。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                二值化图像、灰度图像和处理后的原始帧。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        if prev_frame is None:
            return bin_img, gray, frame
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray, prev_gray)
        _, frame_diff = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = self.filter_contours(contours)
        self.update_trajectories(candidates)
        
        return bin_img, gray, frame

    def filter_contours(self, contours: List[np.ndarray]) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """
        过滤轮廓，选择可能的飞镖候选。

        Args:
            contours (List[np.ndarray]): 从帧差分中检测到的轮廓列表。

        Returns:
            List[Tuple[Tuple[float, float], Tuple[float, float], float]]: 
                符合条件的候选轮廓列表，每个元素是 minAreaRect 的结果。
        """
        candidates = []
        for contour in contours:
            if len(contour) < 5:
                continue
            rect = cv2.minAreaRect(contour)
            if self.is_valid_rect(rect):
                candidates.append(rect)
        return candidates

    def is_valid_rect(self, rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> bool:
        """
        检查矩形是否符合飞镖的大小和形状条件。

        Args:
            rect (Tuple[Tuple[float, float], Tuple[float, float], float]): 
                表示矩形的元组，包含中心点坐标、尺寸和旋转角度。

        Returns:
            bool: 如果矩形符合条件则返回 True，否则返回 False。
        """
        _, (w, h), _ = rect
        if w < 3 or w > 80 or h < 3 or h > 80:
            return False
        if (w < h and w * 3 < h) or (h < w and h * 3 < w):
            return False
        return True

    def update_trajectories(self, candidates: List[Tuple[Tuple[float, float], Tuple[float, float], float]]):
        """
        更新轨迹列表，匹配新的候选点到现有轨迹或创建新轨迹。

        Args:
            candidates (List[Tuple[Tuple[float, float], Tuple[float, float], float]]): 
                候选点列表，每个元素是 minAreaRect 的结果。
        """
        new_trajectories = []
        matched_trajectories = [False] * len(self.trajectories)
        
        for candidate in candidates:
            best_match = self.find_best_match(candidate, matched_trajectories)
            if best_match is not None:
                traj_index, _ = best_match
                new_traj = Trajectory(self.trajectories[traj_index].points + [candidate[0]], 5)
                new_trajectories.append(new_traj)
                matched_trajectories[traj_index] = True
            else:
                new_trajectories.append(Trajectory([candidate[0]], 5))
        
        self.update_unmatched_trajectories(new_trajectories, matched_trajectories)
        self.trajectories = new_trajectories

    def find_best_match(self, candidate: Tuple[Tuple[float, float], Tuple[float, float], float], matched_trajectories: List[bool]) -> Tuple[int, float]:
        """
        为给定的候选点找到最佳匹配的现有轨迹。

        Args:
            candidate (Tuple[Tuple[float, float], Tuple[float, float], float]): 
                候选点，minAreaRect 的结果。
            matched_trajectories (List[bool]): 标记已匹配轨迹的列表。

        Returns:
            Tuple[int, float]: 最佳匹配轨迹的索引和匹配误差。如果没有匹配，返回 None。
        """
        best_match = None
        least_err = float('inf')
        
        for j, traj in enumerate(self.trajectories):
            if matched_trajectories[j]:
                continue
            
            if not self.is_valid_match(traj, candidate):
                continue
            
            if len(traj.points) >= self.config['fit_pts_num']:
                err = self.calculate_fit_error(traj, candidate)
                if err < least_err:
                    best_match = (j, err)
                    least_err = err
            else:
                return j, 0
        
        return best_match
   
    def is_valid_match(self, traj: Trajectory, candidate: Tuple[Tuple[float, float], Tuple[float, float], float]) -> bool:
        """
        检查候选点是否与给定轨迹匹配。

        Args:
            traj (Trajectory): 现有轨迹。
            candidate (Tuple[Tuple[float, float], Tuple[float, float], float]): 
                候选点，minAreaRect 的结果。

        Returns:
            bool: 如果匹配有效则返回 True，否则返回 False。
        """
        if len(traj.points) < 2:
            return True
        
        last_point = traj.points[-1]
        second_last_point = traj.points[-2]
        candidate_point = candidate[0]
        
        motion = np.array(last_point) - np.array(candidate_point)
        dist = np.linalg.norm(motion)
        
        if dist > self.config['max_dist']:
            return False
        
        if (last_point[0] < candidate_point[0] and second_last_point[0] > last_point[0]) or \
           (last_point[0] > candidate_point[0] and second_last_point[0] < last_point[0]):
            return False
        
        dir_prev = np.array(last_point) - np.array(second_last_point)
        dir_curr = np.array(candidate_point) - np.array(last_point)
        
        if np.allclose(dir_prev, 0) or np.allclose(dir_curr, 0):
            return False
        
        cos_angle = dir_prev.dot(dir_curr) / (np.linalg.norm(dir_prev) * np.linalg.norm(dir_curr))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.arccos(cos_angle) <= np.pi / 4

    def calculate_fit_error(self, traj: Trajectory, candidate: Tuple[Tuple[float, float], Tuple[float, float], float]) -> float:
        """
        计算候选点与轨迹拟合曲线的误差。

        Args:
            traj (Trajectory): 现有轨迹。
            candidate (Tuple[Tuple[float, float], Tuple[float, float], float]): 
                候选点，minAreaRect 的结果。

        Returns:
            float: 拟合误差。如果误差超过阈值，返回无穷大。
        """
        points = traj.points[-self.config['fit_pts_num']:] + [candidate[0]]
        coeffs = self.fit_points(points)
        
        errors = [abs(self.quadratic(point[0], *coeffs) - point[1]) for point in points]
        if max(errors) > self.config['traj_error_limit']:
            return float('inf')
        
        return abs(self.quadratic(candidate[0][0], *coeffs) - candidate[0][1])

    def update_unmatched_trajectories(self, new_trajectories: List[Trajectory], matched_trajectories: List[bool]):
        """
        更新未匹配的轨迹，减少它们的生命值或删除它们。

        Args:
            new_trajectories (List[Trajectory]): 新的轨迹列表，将被更新。
            matched_trajectories (List[bool]): 标记已匹配轨迹的列表。
        """
        for j, matched in enumerate(matched_trajectories):
            if not matched:
                traj = self.trajectories[j]
                traj.lifetime -= 1
                if traj.lifetime > 0:
                    new_trajectories.append(traj)

    def visualize_trajectories(self, frame: np.ndarray):
        """
        在帧上可视化所有轨迹。

        Args:
            frame (np.ndarray): 要绘制轨迹的视频帧。
        """
        for traj in self.trajectories:
            if len(traj.points) <= 1:
                continue
            color = (0, 255, 0) 

            # 使用原有的 draw_trajectory 方法，保留透明度设置
            self.draw_trajectory(frame, traj.points, color)

        if self.trajectories:
            longest_traj = max(self.trajectories, key=lambda t: len(t.points))
            if len(longest_traj.points) > 1:
                # 以半透明方式突出显示最长轨迹
                self.draw_trajectory(frame, longest_traj.points, (0, 255, 0), 0.4)

            if len(longest_traj.points) > self.config['fit_pts_num']:
                # 为最长轨迹添加预测曲线
                coeffs = self.fit_points(longest_traj.points)
                direction = 1 if longest_traj.points[0][0] < longest_traj.points[1][0] else -1
                self.draw_curve(frame, coeffs, (255, 0, 255), int(longest_traj.points[-1][0]), direction)

        # 如果最长轨迹足够长，只保留最长轨迹
        if self.trajectories and len(longest_traj.points) > 6:
            self.trajectories = [longest_traj]

    def run(self):
        """
        运行飞镖检测器的主循环。

        这个方法处理视频的每一帧，检测飞镖轨迹，并将结果可视化。
        处理后的视频被保存到输出文件。
        """
        prev_frame = None
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.detect_light_change(frame):
                print("Light change detected. Resetting trajectories.")
                self.trajectories = []
            
            bin_img, gray, frame = self.process_frame(frame, prev_frame)
            
            self.visualize_trajectories(frame)
            
            self.trajectories = [traj for traj in self.trajectories if not self.is_point_out_of_frame(traj.points[-1])]
            
            cv2.imshow("Frame", frame)
            self.writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            prev_frame = frame.copy()
        
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DartDetector("./data/task1.avi", "./data/task1_result.avi")
    # detector = DartDetector("./data/task2.avi", "./data/task2_result.avi")
    detector.run()