# 飞镖轨迹检测与跟踪系统

## 1. 概述

本程序实现了一个飞镖轨迹检测与跟踪系统，能够从视频中识别飞镖的飞行轨迹，并进行可视化展示。该系统主要用于分析飞镖的飞行路径，可应用于飞镖运动的训练和分析中。

## 2. 主要功能

1. 从视频中检测飞镖的运动
2. 跟踪并记录飞镖的飞行轨迹
3. 对飞行轨迹进行拟合和预测
4. 可视化展示飞镖轨迹
5. 处理光照变化等干扰因素

## 3. 核心类和方法

### 3.1 Trajectory 类

表示飞镖轨迹的数据类。

属性：
- `points`: 轨迹点列表，每个点是 (x, y) 坐标
- `lifetime`: 轨迹的剩余生命值

### 3.2 DartDetector 类

飞镖检测器的主类，包含所有核心功能。

主要属性：
- `cap`: 视频捕获对象
- `writer`: 视频写入对象
- `config`: 配置参数字典
- `trajectories`: 当前跟踪的轨迹列表
- `last_light_level`: 上一帧的平均亮度

主要方法：

#### 3.2.1 `__init__(self, video_file: str, output_file: str)`
初始化检测器，设置输入输出视频文件。

#### 3.2.2 `quadratic(x: float, a: float, b: float, c: float) -> float`
静态方法，定义二次函数。

#### 3.2.3 `fit_points(self, points: List[Tuple[float, float]]) -> Tuple[float, float, float]`
使用最小二乘法拟合点集到二次函数。

#### 3.2.4 `draw_trajectory(self, img: np.ndarray, traj: List[Tuple[float, float]], color: Tuple[int, int, int], alpha: float = 1)`
在图像上绘制轨迹。

#### 3.2.5 `draw_curve(self, img: np.ndarray, coeffs: Tuple[float, float, float], color: Tuple[int, int, int], start_x: int, direction: int, alpha: float = 1)`
在图像上绘制二次曲线。

#### 3.2.6 `is_point_out_of_frame(self, point: Tuple[float, float]) -> bool`
检查点是否超出图像边界。

#### 3.2.7 `detect_light_change(self, frame: np.ndarray) -> bool`
检测图像的光照变化。

#### 3.2.8 `process_frame(self, frame: np.ndarray, prev_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
处理单个视频帧，进行运动检测和轨迹更新。

#### 3.2.9 `filter_contours(self, contours: List[np.ndarray]) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]`
过滤轮廓，选择可能的飞镖候选。

#### 3.2.10 `is_valid_rect(self, rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> bool`
检查矩形是否符合飞镖的大小和形状条件。

#### 3.2.11 `update_trajectories(self, candidates: List[Tuple[Tuple[float, float], Tuple[float, float], float]])`
更新轨迹列表，匹配新的候选点到现有轨迹或创建新轨迹。

#### 3.2.12 `find_best_match(self, candidate: Tuple[Tuple[float, float], Tuple[float, float], float], matched_trajectories: List[bool]) -> Tuple[int, float]`
为给定的候选点找到最佳匹配的现有轨迹。

#### 3.2.13 `is_valid_match(self, traj: Trajectory, candidate: Tuple[Tuple[float, float], Tuple[float, float], float]) -> bool`
检查候选点是否与给定轨迹匹配。

#### 3.2.14 `calculate_fit_error(self, traj: Trajectory, candidate: Tuple[Tuple[float, float], Tuple[float, float], float]) -> float`
计算候选点与轨迹拟合曲线的误差。

#### 3.2.15 `update_unmatched_trajectories(self, new_trajectories: List[Trajectory], matched_trajectories: List[bool])`
更新未匹配的轨迹，减少它们的生命值或删除它们。

#### 3.2.16 `visualize_trajectories(self, frame: np.ndarray)`
在帧上可视化所有轨迹。

#### 3.2.17 `run(self)`
运行飞镖检测器的主循环。

## 4. 核心算法

### 4.1 飞镖检测
- 使用帧差法检测运动物体
- 应用自适应阈值进行图像二值化
- 轮廓检测和筛选，根据大小和形状特征识别可能的飞镖

### 4.2 轨迹跟踪
- 使用最近邻匹配算法将新检测到的点与现有轨迹匹配
- 维护轨迹的生命周期，删除不活跃的轨迹

### 4.3 轨迹拟合和预测
- 使用最小二乘法将轨迹点拟合到二次函数
- 基于拟合的函数预测飞镖的未来位置

### 4.4 光照变化检测
- 监控帧间平均亮度的变化
- 当检测到显著亮度变化时，重置轨迹跟踪

## 5. 配置参数

系统包含多个可调参数，存储在 `config` 字典中：

- `max_dist`: 两点间最大允许距离
- `fit_pts_num`: 用于拟合的点数
- `traj_error_limit`: 轨迹拟合误差限制
- `light_change_threshold`: 光照变化阈值
- `edge_threshold`: 边缘检测阈值
- `min_traj_length`: 最小轨迹长度
- `max_traj_lifetime`: 最大轨迹生命周期

## 6. 输入输出

### 6.1 输入
- 视频文件路径

### 6.2 输出
- 处理后的视频文件，包含可视化的轨迹
- 实时显示处理后的每一帧

## 7. 使用方法

1. 实例化 `DartDetector` 类，提供输入视频和输出视频的文件路径
2. 调用 `run()` 方法开始处理

示例：
```python
detector = DartDetector("./data/input_video.avi", "./data/output_video.avi")
detector.run()
```

## 8. 性能考虑

- 使用 NumPy 进行高效的数组操作
- 采用帧差法减少计算量
- 轨迹筛选和合并策略减少误检

## 9. 限制和潜在改进

- 当前系统假设飞镖轨迹近似二次函数，可能不适用于所有情况
- 光照变化检测可能需要更复杂的算法来处理gradual变化
- 可以考虑添加机器学习模型来提高飞镖识别的准确性
- 多目标跟踪能力有限，可以考虑实现更复杂的数据关联算法

## 10. 依赖库

- OpenCV (cv2)
- NumPy
- SciPy

## 11. 注意事项

- 确保输入视频格式兼容
- 程序假设视频中主要运动物体为飞镖，复杂背景可能导致误检
- 处理高分辨率视频可能需要较高的计算资源
