# 工业场景纯视觉避障系统 v2.1

## 项目简介

隧道/工业场景下的**纯视觉实时避障系统**。基于 OpenCV 传统视觉，在单目摄像头输入下实现车道边界检测、障碍物感知、路径规划与安全降级的完整管线。

核心流程：摄像头取流 → ROI 裁剪 → CLAHE 增强 → Canny 边缘 → **车道边界检测（隔离沟+隔离带）** → 可行驶区域评分 → **碎石检测** → 障碍检测 → **DWA 路径规划** → 避障决策 → **安全降级** → 车辆控制。

v2.1 新增 Phase 1 模块（车道边界、碎石、路径规划、标定），并修复 8 个生产级安全问题。

## 运行环境

- Python 3.8+
- Linux（推荐）/ Windows
- USB 摄像头或工业相机

## 安装

```bash
cd vision_obstacle_avoidance
pip install -r requirements.txt
```

## 运行

```bash
# 摄像头实时测试
python main.py

# 视频文件测试（修改 config.py 中 CAMERA_INDEX = "test_video.mp4"）
python main.py

# 按 q 键退出
```

## 架构：5 层管线 + 安全降级

```
摄像头(0) → 预处理(1) → 感知层(2) → 占用栅格(3) → 路径规划+决策(4) → 安全降级(5) → 控制
```

### 第 0 层 — 图像获取
- `camera_capture.py` — USB/工业相机采集，断线自动重连（指数退避）
- 支持视频文件输入

### 第 1 层 — 预处理
- `preprocess.py` — ROI 裁剪 + CLAHE 增强 + Canny 边缘 + HSV 地面分割
- `exposure_detector.py` — 过曝/欠曝检测（连续帧确认防抖）
- `temporal_denoiser.py` — 时域降噪（3 帧加权滑动窗口）
- `dehazer.py` — 暗通道先验去雾

### 第 2 层 — 感知（CV + 可选 DL 双轨）
**传统 CV 轨：**
- `lane_boundary_detector.py` — **v2.1** 隔离沟列投影检测 + 隔离带 HoughLinesP 聚类 + 跳变抑制
- `lane_or_freespace_detector.py` — 三区可行驶区域评分，优先动态车道边界
- `obstacle_detector.py` — 轮廓检测 + 面积/宽高比过滤 + 三区归类 + 危险等级
- `debris_detector.py` — **v2.1** 碎石/碎渣边缘密度异常检测（网格分析）
- `calibration.py` — **v2.1** 消失点累积标定（RANSAC），像素→度量转换

**深度学习轨（可选）：**
- `hybridnets_engine.py` — HybridNets ONNX 推理，失败自动回退到 CV

### 第 3 层 — 占用栅格
- `occupancy_grid.py` — 多源融合（分割 + 边界 + 检测框 + 碎石）

### 第 4 层 — 规划 + 决策
- `path_planner.py` — **v2.1** DWA 路径规划（阿克曼非线性转向 + dt限幅 + 向量化碰撞 + 精细化占用）
- `decision_maker.py` — **v2.1** 规则引擎（动态隔离沟方向 + 4 级驾驶状态 + 连续帧防抖）

### 第 5 层 — 安全降级
- `safety_degrader.py` — **v2.1** 5 级安全状态机（迟滞恢复阈值 + L3超时 + 逐级恢复）

## 模块一览

| 文件 | 功能 | 版本 |
|------|------|:--:|
| `main.py` | 主循环，9 步帧处理 + FPS 统计 | v1.0 |
| `config.py` | ~100 个可调参数，分 11 组 | v2.1 |
| `camera_capture.py` | 摄像头采集，断线重连 | v1.0 |
| `preprocess.py` | 6 步预处理管线 | v1.0 |
| `exposure_detector.py` | 过曝/欠曝检测 | v1.0 |
| `temporal_denoiser.py` | 时域降噪 | v1.0 |
| `dehazer.py` | 暗通道去雾 | v1.0 |
| `calibration.py` | 消失点累积标定 | Phase 1 |
| `lane_boundary_detector.py` | 隔离沟 + 隔离带检测 | Phase 1 / v2.1 |
| `lane_or_freespace_detector.py` | 三区自由空间评分 | v1.0 / Phase 1 改 |
| `obstacle_detector.py` | 轮廓障碍检测 | v1.0 |
| `debris_detector.py` | 碎石边缘密度异常检测 | Phase 1 |
| `occupancy_grid.py` | 多源占用栅格融合 | Phase 1 |
| `path_planner.py` | 车道内 DWA 路径规划 | Phase 1 / v2.1 |
| `decision_maker.py` | 10 规则引擎 + 状态机 | v1.0 / Phase 1 改 / v2.1 |
| `safety_degrader.py` | 5 级安全降级状态机 | Phase 1 / v2.1 |
| `vehicle_controller.py` | 车辆控制接口（占位） | v1.0 |
| `logger.py` | CSV + 视频日志 | v1.0 |
| `utils.py` | 7 个 dataclass + 可视化 | v1.0 / Phase 1 改 |

## v2.1 修复清单

| # | 严重度 | 问题 | 文件 | 修复 |
|:--:|:--:|------|------|------|
| 1 | 🔴 | 安全降级无迟滞，置信度震荡导致急刹/全速抽搐 | `safety_degrader.py` | 恢复阈值 0.70/0.50/0.30 > 降级阈值，需连续 10 帧确认 |
| 2 | 🔴 | L3 计时器错误重置，震荡时 L4 永不触发 | `safety_degrader.py` | 只在 L0 恢复时重置 `_l3_entry_time` |
| 3 | 🔴 | 阿克曼转向线性映射，大转角偏离物理模型 | `path_planner.py` | `atan(k × wheelbase) / max_steer_rad` 非线性映射 |
| 4 | 🟠 | 帧间转向限幅无时间绑定，FPS 波动导致速率失控 | `path_planner.py` | `max_delta = 60°/s × π/180 × dt` |
| 5 | 🟠 | 障碍投影过于粗暴，轻度阻塞封死 20-80% 车道 | `path_planner.py` | 三级占用：全阻塞/部分占用(缩范围)/不占用 |
| 6 | 🟠 | Python 嵌套循环碰撞检测 O(N×M×K) | `path_planner.py` | `_dist_to_rects_vectorized()` numpy 批量 |
| 7 | 🔴 | 隔离沟方向硬编码，反向隧道直接掉沟 | `decision_maker.py` | 动态判断 ditch_px 与两侧隔离带位置关系 |
| 8 | 🟡 | 时序滤波缺跳变抑制，暗色井盖/水渍污染滤波器 | `lane_boundary_detector.py` | `_innovation_check()` 偏离中位数 > 50px 拒绝 |

## 参数调节

所有可调参数集中在 `config.py`，按功能分组：

| 参数组 | 关键参数 | 说明 |
|--------|---------|------|
| 摄像头 | `CAMERA_INDEX`, `FRAME_WIDTH`, `FRAME_HEIGHT` | 设备索引和分辨率 |
| ROI | `ROI_TOP_RATIO` ~ `ROI_RIGHT_RATIO` | 分析区域边界 |
| 预处理 | `CANNY_LOW/HIGH`, `CLAHE_CLIP_LIMIT`, `GROUND_HSV_*` | 边缘/光照/地面分割 |
| 标定 | `CAMERA_HEIGHT_M`, `CAMERA_PITCH_DEG`, `VP_*` | 相机几何和消失点 |
| 车道边界 | `DITCH_*`, `BARRIER_*`, `LINE_*`, `BOUNDARY_*` | 隔离沟/隔离带检测 |
| 碎石 | `DEBRIS_GRID_CELL`, `DEBRIS_LARGE_CM`, `DEBRIS_*_STD_MULT` | 碎石检测参数 |
| 路径规划 | `PATH_NUM_CURVATURES`, `PATH_MAX_CURVATURE`, `PATH_CLEARANCE_MIN_CM` | DWA 候选和约束 |
| 转向 | `STEERING_*`, `STEERING_MAX_RATE_DEG_PER_SEC`, `STEERING_MAX_ANGLE_DEG` | v2.1 新增 |
| 决策 | `DECISION_CONSECUTIVE_FRAMES`, `CENTER_OFFSET_THRESHOLD` | 防抖和灵敏度 |
| 调试 | `DEBUG_VIEW`, `SAVE_LOG`, `SAVE_VIDEO` | 可视化开关 |

## 控制指令

| 指令 | 含义 | 触发条件 |
|------|------|---------|
| FORWARD | 直行 | 中心畅通 |
| TURN_LEFT | 左转 | 中心阻塞，左侧畅通 |
| TURN_RIGHT | 右转 | 中心阻塞，右侧畅通 |
| STOP | 停车 | 三区全阻塞 / 车道不可通行 / 越沟检测 / 感知丢失 |
| SLOW_DOWN | 减速 | 可行驶区域无效 / 延迟超标 / 安全降级 |

## 当前限制

- 不包含深度学习模型管线（预留 hybridnets_engine 接口）
- 不包含多传感器融合（无激光雷达/超声波/IMU）
- 控制接口为占位实现，需接入真实底盘协议（CAN/串口）
- 标定精度依赖相机安装高度和俯仰角的实测值
- 缺少真实隧道场景视频验证隔离沟/隔离带检测

## 后续方向

- Phase 2: YOLOv8 + ByteTrack 动态障碍检测跟踪 / 极端光照处理 / 标准 DWA 升级
- Phase 3: 恶劣天气 / 传感器退化 / Fail-Safe 硬件安全链路
- 多线程流水线并行（采集/检测/决策分离）
- 控制执行层：Pure Pursuit 路径跟踪 + PID 速度闭环
- IMU + 轮速计融合定位
- 单元测试 / 集成测试 / 仿真验证（CARLA/Gazebo）
