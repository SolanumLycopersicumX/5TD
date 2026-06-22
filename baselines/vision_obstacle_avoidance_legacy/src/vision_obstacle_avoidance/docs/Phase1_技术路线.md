# 隧道施工场景纯视觉避障 — 算法与系统文档

## 第一章：系统总览

### 1.1 项目文件组织

项目根目录下共有 20 个源文件。按功能分为四组。

**管道核心（每帧执行的避障逻辑）：**

| 文件 | 模块 | 职责 |
|------|------|------|
| `main.py` | 主循环 | 串联全部模块的帧循环，FPS 统计，可视化调试面板 |
| `config.py` | 参数配置 | 所有可调参数集中管理，约 60 个参数 |
| `camera_capture.py` | `CameraCapture` | 摄像头取流，断线重连，letterbox 缩放 |
| `preprocess.py` | `ImagePreprocessor` | ROI 裁剪、CLAHE 增强、Canny 边缘、HSV 地面分割 |
| `lane_or_freespace_detector.py` | `FreeSpaceDetector` | 三区自由空间评分，Phase 1 扩展支持动态车道边界 |
| `obstacle_detector.py` | `ObstacleDetector` | 轮廓障碍检测，面积/宽高比过滤，危险等级计算 |
| `decision_maker.py` | `DecisionMaker` | 规则引擎 + 状态机，Phase 1 扩展支持硬边界约束 |
| `vehicle_controller.py` | `VehicleController` | 控制输出占位（预留串口/CAN/ROS 接口） |
| `logger.py` | `RuntimeLogger` | CSV 日志记录 + 视频保存 |
| `utils.py` | 数据类 + 可视化 | 7 个 dataclass + 4 个绘制函数 |

**Phase 1 新增模块：**

| 文件 | 模块 | 职责 |
|------|------|------|
| `calibration.py` | `GroundCalibrator` | 消失点估计 → 像素到地面平面映射 |
| `lane_boundary_detector.py` | `LaneBoundaryDetector` | 隔离沟列投影检测 + 隔离带 HoughLinesP 聚类 |
| `debris_detector.py` | `DebrisDetector` | 网格边缘密度异常检测 → 碎石定位与分类 |
| `path_planner.py` | `PathPlanner` | 简化 DWA：可通行性预检 + 离散曲率评估 + 三因子评分 |

**辅助工具（不参与主循环）：**

| 文件 | 说明 |
|------|------|
| `analyze_test.py` | 离线视频分析，输出 JSON 报告 |
| `hardware_discovery.py` | Linux 摄像头/麦克风自动发现 |
| `transcribe.py` | 语音转文字交互工具 |
| `voice_daemon.py` | 语音守护进程（SIGUSR1 触发） |
| `voice_gui.py` | 语音输入 Tkinter GUI |
| `voice_hotkey.py` | 全局热键语音输入 |

### 1.2 帧循环完整流程

每一帧的处理顺序如下。标注"现有"的模块为 Phase 1 未改动，"改"为修改过但保持向后兼容，"新"为 Phase 1 新增。

**步骤 1：取流。** `CameraCapture.read()` 从摄像头或视频文件读取一帧，返回 BGR 格式的 numpy 数组和时间戳。如果配置了 `PRESERVE_ASPECT_RATIO=True`，使用 letterbox 方式缩放（保持源宽高比，灰边填充），否则使用暴力拉伸。摄像头断线时自动按指数退避重连，超过最大重试次数后返回 None（触发决策层 STOP）。（现有，改）

**步骤 2：预处理。** `ImagePreprocessor.process(frame)` 对帧做六步处理：按配置的 ROI 比例裁剪 → 灰度化 → CLAHE 自适应直方图均衡化 → 高斯滤波 → Canny 边缘检测 → HSV 地面分割 + 形态学开闭运算。输出 `PreprocessResult`，包含 `roi_frame`、`gray`、`enhanced`、`edges`、`binary_mask` 和 `debug_images`。（现有，未改）

**步骤 2b：消失点累积。** `GroundCalibrator.estimate_vp_from_edges()` 从 Canny 边缘图中提取纵向 Hough 直线，用 RANSAC 求消失点。`accumulate()` 将消失点加入累积窗口，当窗口内方差收敛时标定完成。此步骤非阻塞——标定未收敛时模块返回 None，下游自动回退。（新）

**步骤 3：车道边界检测。** `LaneBoundaryDetector.detect()` 同时执行两个子检测。隔离沟检测（`_detect_ditch`）：取 ROI 下半部 HSV 的 V 通道做列投影，在路面中间区域搜索亮度局部最小值，验证沿该位置的纵向边缘密度。隔离带检测（`_detect_barriers`）：对边缘图运行 HoughLinesP，筛选纵向角度，用消失点约束过滤，底部 x 坐标左右聚类。两个结果各自经十帧中值滤波输出。返回 `LaneBoundaryState`。（新）

**步骤 3b：自由空间评分。** `FreeSpaceDetector.detect()` 沿用三区评分逻辑。当 `lane_boundary` 参数有效时，三区边界使用检测到的实际隔离带和隔离沟位置而非固定比例。隔离沟对面的区域评分设为零。当 `lane_boundary` 无效时，完全回退到原有固定分区行为。（现有，改）

**步骤 4：障碍检测。** `ObstacleDetector.detect()` 在 Canny 边缘图上找轮廓，过滤面积和宽高比异常值，按左中右分区归类，计算危险等级（0.5×底部接近度 + 0.5×面积比例）。返回 `ObstacleState`。（现有，未改）

**步骤 4b：碎石检测。** `DebrisDetector.detect()` 在车道边界内的边缘图上划分 20×20 网格，计算每单元边缘密度，用 3×3 滑动窗口统计局部均值和标准差，标记异常单元为碎石斑块。如果标定可用，将像素尺寸换算为厘米并分类（<5cm 忽略，5-15cm 记录，>15cm 标记为需避让）。返回 `DebrisState`。（新）

**步骤 4c：路径规划。** `PathPlanner.plan()` 首先检查车道宽度是否足够（车道宽 ≥ 车身宽 + 2×安全余量）。然后生成七条候选圆弧轨迹，每条模拟前向滚动五米，分别评估安全间距（权重 50%）、转向平滑性（30%）、前进进度（20%）。选总分最高的轨迹。若无轨迹满足最小安全间距，输出不可通行。返回 `PathPlan`。（新）

**步骤 5：决策。** `DecisionMaker.decide()` 依次评估九条规则（Phase 1 新增两条在最前面）。规则 0：路径不可通行 → STOP。规则 0.5：检测到跨越隔离沟倾向 → STOP。规则 -1：感知输入丢失 → STOP。规则 1：延迟/FPS 超标 → SLOW_DOWN。规则 2：可行驶区域无效 → SLOW_DOWN/STOP。规则 3：三区全阻塞 → STOP。规则 4：中心阻塞 → 动态绕行。规则 5：偏离中心 → tanh 微调。规则 6：畅通 → 直行。输出 `Decision`。所有新增参数均为可选，默认 None 时规则 0 和 0.5 不触发。（现有，改）

**步骤 6：控制输出。** `VehicleController.send(decision)` 输出转向角和速度指令。当前为占位实现（控制台打印），预留了串口/CAN/ROS 的接口注释。（现有，未改）

**步骤 7：日志。** `RuntimeLogger.update()` 将每帧的 FPS、延迟、决策、感知状态、以及 Phase 1 新增的车道边界、碎石、路径规划字段写入 CSV。可选地保存标注视频。（现有，改）

**步骤 8：FPS 计算。** 每 30 帧统计一次。（现有，未改）

**步骤 9：可视化。** 若 `DEBUG_VIEW=True`，`build_debug_display()` 组装多面板调试窗口。Phase 1 扩展后，下排包含 FreeSpace、Obstacle、LaneBoundary、PathPlan、Decision 五个面板。（现有，改）

### 1.3 数据结构

系统定义了七个 dataclass，全部在 `utils.py` 中。

**`PreprocessResult`** — 预处理输出。包含 `roi_frame`（BGR 裁剪图）、`gray`（灰度图）、`enhanced`（CLAHE 增强图）、`edges`（Canny 边缘二值图）、`binary_mask`（HSV 地面分割二值图）、`debug_images`（中间步骤的可视化字典，键名为 "0_original" 到 "6_mask"）。

**`FreeSpaceState`** — 可行驶区域分析结果。`is_valid` 表示分析是否可靠（置信度 ≥ 0.4）。三个评分 `left_free_score`、`center_free_score`、`right_free_score` 各自在 0 到 1 之间，综合了边缘密度（30% 权重）和 HSV 地面 mask 覆盖率（70% 权重）。`center_offset` 是车辆相对车道中心的偏移，正值表示右侧更畅通（车辆偏左了），负值表示左侧更畅通。`confidence` 取中心区域的评分值。

**`ObstacleState`** — 障碍检测结果。`has_obstacle` 表示是否检测到有效障碍物。`obstacle_boxes` 是障碍物边界框列表 `[(x,y,w,h), ...]`。`danger_level` 取所有障碍物中最大的危险评分（0 到 1），评分由底部接近度（越靠画面底部越近）和面积比例各占 50% 合成。三个布尔值 `blocked_left/center/right` 表示各区域是否被障碍物占据。`closest_obstacle_zone` 记录最近障碍物所在区域。

**`Decision`** — 决策输出。`command` 为六种指令之一（FORWARD、TURN_LEFT、TURN_RIGHT、STOP、SLOW_DOWN、SEARCH_LANE）。`speed` 和 `steering` 各自在 0~1 和 -1~1 之间。`reason` 是决策原因文本。`confidence` 和 `drive_state`（NORMAL/CAUTION/EVASIVE/EMERGENCY）随风险升高而升级。

**`LaneBoundaryState`**（Phase 1 新增）— 纵向结构检测结果。三个像素坐标 `left_barrier_px`、`ditch_px`、`right_barrier_px` 分别定位左侧隔离带、中央隔离沟、右侧隔离带在 ROI 中的水平位置，任一可为 None（未检测到）。`is_valid` 至少需要隔离沟被检测到。`vanishing_point` 来自标定模块（如可用）。`lane_width_m` 是估算的米制车道宽度。`confidence` 综合三个结构的检测情况。

**`DebrisState`**（Phase 1 新增）— 碎石检测结果。`has_debris` 表示车道内存在至少一个碎石斑块。`debris_boxes` 列出每个斑块的位置、大小和厘米尺寸。`has_large_debris` 为 True 时表示存在超过 15cm 的大石块需要避让。

**`PathPlan`**（Phase 1 新增）— 路径规划结果。`passable` 为 False 时表示车道不可通行，`blocked_reason` 说明原因（LANE_TOO_NARROW 或 OBSTACLE）。`steering` 和 `speed` 是规划的最优转向角和速度。`clearance_cm` 是规划路径的最小通过间距（厘米）。

---

## 第二章：现有模块算法详述

本章详细描述 Phase 1 未修改或仅做参数扩展的现有模块。已做修改的模块在修改点做了标注。

### 2.1 CameraCapture — 摄像头采集

**初始化：** 选择 V4L2 后端（Linux 实摄像头）或 CAP_ANY（视频文件）。设置目标分辨率和帧率。如果启用断线重连，仅在 Linux 且非视频文件时生效。调用 `_open_camera()` 打开设备。

**`read()`：** 若设备已断开（`isOpened()` 返回 False），进入重连逻辑：计算指数退避延迟（`base_delay × 2^attempt`，上限 `max_delay`），未到时间直接返回 None，到了时间尝试 `_open_camera()`。超过最大重试次数后永久放弃，由决策层触发 STOP。

正常读取时调用 `cap.read()` 获取帧，然后调用 `_resize_frame()` 缩放。如果是视频文件且播放到末尾，循环到开头。

**`_resize_frame()`（Phase 1 修改点）：** 如果源分辨率与目标分辨率一致，直接返回。如果 `PRESERVE_ASPECT_RATIO=True`，计算缩放比例使长边适配目标尺寸，居中放在目标大小的灰色画布上。否则使用现有暴力拉伸。

**容错：** 断线重连使用指数退避防止 CPU 空转。重连成功后立即尝试读取一帧，失败则关闭设备等下次重试。

### 2.2 ImagePreprocessor — 图像预处理

**初始化：** 创建 CLAHE 对象（`clipLimit` 控制对比度增强强度，`tileGridSize` 控制局部网格大小）。创建椭圆形态学核。高斯核大小需为奇数，否则抛异常。

**`process(frame)`：** 返回一个 `PreprocessResult`。

1. ROI 裁剪。根据四个比例参数从原帧中切出感兴趣区域。如果 ROI 为空（参数错误导致），返回空的 PreprocessResult。
2. `cv2.cvtColor(roi_frame, COLOR_BGR2GRAY)` 转灰度。
3. `clahe.apply(gray)` 做自适应直方图均衡化。CLAHE 在局部区域内拉伸对比度，隧道场景中暗区的细节被增强，亮区不会被过度拉伸。
4. `cv2.GaussianBlur(enhanced, (5,5), 0)` 高斯滤波。核大小必须是奇数，σ 自动从核大小推算。
5. `cv2.Canny(blurred, 50, 150)` Canny 边缘检测。低阈值 50 以下的不认为是边缘，高阈值 150 以上的确定为边缘，50~150 之间的如果和高阈值边缘相连则保留。这两个值在 `config.py` 中可调。
6. HSV 地面分割。将 ROI 彩色图转 HSV 空间，用 `cv2.inRange` 和可配置的上下界（默认 `[0,0,40]` 到 `[180,60,255]`）提取浅色地面的像素。然后用椭圆核做开运算（去小噪点）和闭运算（填小空洞），迭代次数可配置。

**中间结果：** `debug_images` 字典保存了每个步骤的可视化图，供 `build_debug_display()` 使用。所有灰度图通过 `COLOR_GRAY2BGR` 转为三通道以便拼接显示。

### 2.3 ObstacleDetector — 障碍检测

**`detect(roi_frame, edges, binary_mask)`：** 输入 ROI 彩色图和 Canny 边缘图，binary_mask 参数保留但当前未使用（为后续扩展预留）。

1. `cv2.findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` 在边缘图上找轮廓。`RETR_EXTERNAL` 只取最外层轮廓，避免内部嵌套边缘的干扰。
2. 逐轮廓过滤。计算 `cv2.contourArea(cnt)`，如果不在 `[OBSTACLE_MIN_AREA, OBSTACLE_MAX_AREA]`（默认 500~50000 像素²）之间则丢弃。计算 `cv2.boundingRect(cnt)` 的宽高比，如果不在 `[OBSTACLE_MIN_ASPECT, OBSTACLE_MAX_ASPECT]`（默认 0.2~5.0）之间则丢弃——这一步排除了地面裂缝和反光条等细长假阳性。
3. 区域分类。按轮廓中心的 x 坐标与 `ZONE_LEFT_RATIO`（默认 0.40）和 `ZONE_RIGHT_RATIO`（默认 0.60）比较，将每个有效轮廓归入 LEFT、CENTER 或 RIGHT 区域，设置对应的 `blocked_*` 布尔值。
4. 危险等级计算。每个障碍物的危险等级 = 0.5 × 底部接近度 + 0.5 × 面积比例。底部接近度 = (y+bh) / ROI高度，越靠近画面底部（离车越近）值越大。面积比例 = min(1.0, area / OBSTACLE_AREA_REFERENCE)。取所有障碍物中最大的危险等级作为 `danger_level`。
5. 最近障碍物。在满足 `proximity > 0.3` 的障碍物中找底部位置最靠下的（离车最近），记录其所在区域到 `closest_obstacle_zone`。

**可视化：** 在 debug 图像上绘制区域分割线、障碍物边界框（红色表示 danger > 0.6，黄色表示较低危险）、危险等级标签、红色水平线标记危险区域起点（`DANGER_ZONE_TOP_RATIO`）。

### 2.4 DecisionMaker — 规则引擎与状态机

**`DriveState` 枚举：** 四种行驶状态，优先级 EMERGENCY > EVASIVE > CAUTION > NORMAL。低置信度决策自动向上一级状态升格以增加安全余量。

**内部状态：** `_last_command` 记录上一帧的指令，`_last_steer` 记录上一帧的转向角，`_cmd_queue` 是一个长度固定为 `DECISION_CONSECUTIVE_FRAMES`（默认 3）的 deque，用于连续帧确认防抖。`_lock` 是 `threading.Lock`，保证线程安全。

**`decide()`（Phase 1 修改点）：** 线程安全入口，新增三个可选参数 `debris_state`、`path_plan`、`lane_boundary`，全部默认为 None。内部加锁后调用 `_decide_impl()`。

**`_decide_impl()` 的规则决策链：**

每条规则返回一个 `Decision` 对象。规则按下列优先级排列，先匹配先返回。标（新）的为 Phase 1 新增规则。

- **规则 -2（新）：安全降级触发。** 如果 `degradation_status` 不为 None 且触发接管条件，立即 EMERGENCY STOP。此规则优先级最高，覆盖所有其他规则。
- **规则 -1：空值保护。** 如果 `free_state` 或 `obstacle_state` 为 None（感知模块崩溃），返回 EMERGENCY STOP。
- **规则 0（新）：车道不可通行。** 如果 `path_plan.passable == False`，返回 EMERGENCY STOP，原因是车道宽度不足或全堵。
- **规则 0.5（新）：隔离沟跨越检测。** 如果 `lane_boundary.is_valid` 为 True 且 `_is_crossing_ditch()` 返回 True，返回 EMERGENCY STOP。此方法检查自由空间的 `center_offset` 是否强烈偏向隔离沟方向（超出阈值两倍）且右侧评分显著高于中心（说明算法把对向车道当成可行驶区域）。
- **规则 1：性能下降。** 单帧延迟超过 `MAX_LATENCY_MS`（默认 100ms）或 FPS 低于 `TARGET_FPS`（默认 15）时，按超出比例动态降低速度到 `DEFAULT_SPEED` 和 `SLOW_SPEED` 之间，输出 SLOW_DOWN。
- **规则 2：可行驶区域无效。** 如果 `free_state.is_valid == False`，且上一帧已经 STOP 则继续保持 STOP，否则 SLOW_DOWN 减速。
- **规则 3：三区全阻塞。** `blocked_left && blocked_center && blocked_right` 全部为 True，输出 EMERGENCY STOP。
- **规则 4：中心阻塞 → 动态绕行。** `blocked_center` 且 `danger_level > 0.4` 时，调用 `_pick_detour_direction()` 选择绕行方向。该函数比较左右两侧的自由空间评分，选择未被阻塞且评分更优的一侧（需至少领先 0.05）。绕行转向角 = `STEERING_SMALL + urgency × (STEERING_LARGE - STEERING_SMALL)`，其中 `urgency = danger_level × (选择侧评分 - 未选择侧评分)`。速度随危险等级线性降低。
- **规则 5：偏离中心 → tanh 微调。** 如果 `|center_offset| > CENTER_OFFSET_THRESHOLD`（默认 0.15），用 tanh 函数将偏移映射到平滑的转向角。公式为 `steer = tanh(offset × STEERING_TANH_GAIN) × STEERING_TANH_MAX`。
- **规则 6：畅通。** 所有条件都不触发，正常直行。

**`_make_decision()`：** 统一构造 Decision 对象。三件事：帧间转向限幅（`_clamp_steer_rate()`，限制每帧转向变化不超过 `STEERING_MAX_RATE`）、置信度传导（低置信度升格状态并降速）、防抖处理（`_apply_hysteresis()`）。

**`_apply_hysteresis()`：** 连续帧确认机制。指令不变时直接返回。指令变化时，需要连续 `DECISION_CONSECUTIVE_FRAMES`（默认 3）帧一致才切换，期间保留旧指令但允许 steering 微调。EMERGENCY 级别的指令跳过防抖直接生效。

---

## 第三章：Phase 1 新增模块算法详述

### 3.1 GroundCalibrator — 像素到度量标定

**问题背景。** 单目摄像头捕获的是二维像素坐标。路径规划和可通行性判断需要真实距离——"前方三米"在图像中对应多少像素、"这个障碍物宽度是 50cm 还是 5cm"。从单目图像恢复尺度的经典方法需要已知尺寸的标定物（棋盘格）或已知相机内参。施工隧道场景中两者都没有。

**算法选择。** 采用消失点法。场景中天然存在大量纵向直线——隔离带边缘、隔离沟边缘、隧道壁结构——这些直线在透视投影下汇聚于消失点。消失点的位置编码了相机的俯仰角信息和焦距信息。配合相机安装高度和路面平面假设，可以建立像素行到地面距离的映射。

**初始化。** 从配置读取相机安装高度（米）和俯仰角（度）。创建长度为 `VP_ACCUMULATE_FRAMES`（默认 50）的消失点累积窗口。初始状态 `calibrated = False`。由于没有标定数据，主点假设为图像中心 (320, 240)，焦距通过消失点位置和俯仰角反推。

**`estimate_vp_from_edges(edges, roi_offset_x, roi_offset_y)`：**

第一步，对输入边缘图运行 `cv2.HoughLinesP`，参数来自 `config.HOUGH_*`。如果直线总数少于 `VP_MIN_LINES`（默认 5），返回 None。

第二步，遍历所有直线段，计算每条的倾角 `atan2(y2-y1, x2-x1)`。保留倾角在 `[LINE_ANGLE_MIN, LINE_ANGLE_MAX]`（约 70° 到 110°）之间的纵向直线。将直线坐标从 ROI 坐标系转回原图坐标系（加上 `roi_offset`）。

第三步，用 RANSAC 求所有纵向直线延长线的交点。每次迭代随机选两条直线，用解析公式计算交点。对所有直线计算到该交点的距离，距离小于 `VP_RANSAC_THRESHOLD`（默认 10 像素）的视为内点。迭代 100 次，保留内点数最多的交点。最后用所有内点对应的直线中点坐标的中位数精炼交点位置。

**`accumulate(vp)`：** 将每帧估算的消失点加入 FIFO 队列。当队列满（默认 50 帧）时，计算 x 和 y 坐标的标准差。两个标准差都小于 5 像素时判定为收敛：将队列中位数设为正式消失点，调用 `_compute_focal()` 反推焦距，将 `calibrated` 设为 True。

**`_compute_focal()`：** 基于针孔相机模型和地面平面假设。消失点的 y 坐标满足 `vy = cy - f_y × tan(pitch_rad)`，其中 cy 是主点 y 坐标，pitch_rad 是相机俯仰角。由此反推 `f_y = (cy - vy) / tan(pitch_rad)`。如果俯仰角为零，使用默认值 800 像素。

**`meters_at_row(row, image_height)`：** 计算图像第 row 行每像素对应多少米（水平方向）。基于地面平面假设，该行对应的地面点的深度为 `depth = camera_height / tan(pitch_rad + alpha)`，其中 `alpha = atan2(row - cy, fy)` 是像素仰角。然后该行的水平米制分辨率 = `1.0 / (fy / (depth × cos(alpha)))`。如果计算的俯角为负（看向地平线以上），返回 None。

**`pixel_to_ground(x_px, y_px, ...)`：** 将单个像素坐标转换为地面坐标系中的 (x_m, y_m)。横向距离 = (像素 x - 主点 x) × 该行每像素米数。纵向距离 = 该行的地面深度。原点为车辆正下方地面投影点。

**容错设计。** 整个标定模块设计为非阻塞。未收敛时 `calibrated = False`，所有米制查询（`meters_at_row`、`pixel_to_ground`）返回 None。调用方（`LaneBoundaryDetector`、`DebrisDetector`、`PathPlanner`）在获取到 None 时各自回退到像素级判断。这意味着系统在标定完成前仍然可以运行，只是缺少米制精度。直线不足、RANSAC 找不到交点、方差未收敛等情况全部返回 None，不会抛异常。

**局限。** 标定精度依赖于两个关键预设的准确性：相机安装高度和俯仰角。这两个值目前在 config 中为估算值，需要在实车上测量后更新。此外，路面平面假设在坡度或颠簸路段会引入系统误差。消失点的稳定性也依赖于场景中纵向直线的数量和质量——在缺乏纵向结构的空旷场景中可能长期无法收敛。

### 3.2 LaneBoundaryDetector — 纵向结构检测

**问题背景。** 隧道场景中有两种不可跨越的纵向结构。中央隔离沟是路面中间的凹陷排水槽，视觉上表现为比周围路面更暗的纵向条带。两侧隔离带是靠近隧道壁的凸起路缘，视觉上表现为亮度/高度突变导致的纵向边缘簇。两者都是车道边界的硬约束——隔离沟内侧是车道边界，隔离带内侧也是车道边界，小车只能在两者之间的半幅路内行驶。

**初始化。** 维护三个 `deque`，每个长度为 `BOUNDARY_HISTORY_FRAMES`（默认 10），分别用于隔离沟、左隔离带、右隔离带检测结果的时序中值滤波。

**`detect(roi_frame, edges, enhanced, calibrator)`：** 主入口。做空值检查后，依次调用 `_detect_ditch()` 和 `_detect_barriers()`。如果标定器可用且已收敛，获取消失点供隔离带检测使用。将三个子检测的当帧结果分别追加到各自的时序缓冲，取中位数作为最终输出。计算置信度：满分为 1.0，未检测到隔离沟扣 0.4，未检测到左隔离带扣 0.3，未检测到右隔离带扣 0.3。`is_valid` 仅要求隔离沟被检测到（核心约束）。如果标定可用且左侧隔离带和隔离沟都已定位，用标定器的 `meters_at_row()` 估算米制车道宽度。最后调用 `_draw_boundaries()` 在 debug 图像上绘制结构线。

**`_detect_ditch(roi_frame, edges, w, h)`：**

这是 Phase 1 最关键的检测函数。算法基于隔离沟的两个视觉特征：比周围暗（暗色条带）和纵向连续（不是偶然暗斑）。

第一步，取 ROI 的下半部分。`DITCH_STRIP_RATIO_BOTTOM`（默认 0.70）决定了从 ROI 高度的 70% 处到底部的范围。选择下半部是因为近处像素分辨率高、透视变形小。

第二步，将条带转 HSV 空间取 V 通道（明度）。对 V 通道的每一列求平均值，得到一条亮度随水平位置变化的曲线 `col_means`。如果路面平坦均匀，这条曲线应该是平缓的；如果中间有一道暗槽，曲线会出现一个凹陷。

第三步，确定搜索范围。`DITCH_SEARCH_LEFT`（默认 0.20）和 `DITCH_SEARCH_RIGHT`（默认 0.50）限定了隔离沟只可能在 ROI 水平方向的中间偏左区域。这个范围是假设摄像头装在车头偏左位置、隔离沟在车的右侧不远。

第四步，用移动平均（核宽度为 `max(5, w/30)`）平滑列投影曲线，消除像素噪声。在搜索范围内找最小值的位置和值。计算搜索范围内的中位数亮度。如果中位数减最小值的差小于 `DITCH_V_DARK_THRESHOLD`（默认 30 V 值单位），说明这个"暗"只是正常波动，不是真的隔离沟，返回 None。

第五步，验证纵向连续性。在候选 x 坐标 ±5 像素的窄带内，计算从 `strip_top` 到底部的边缘密度。密度低于 `DITCH_EDGE_DENSITY_MIN`（默认 0.05）的说明这个位置的纵向边缘太稀疏，可能是地面污渍而不是结构凹槽，返回 None。

**`_detect_barriers(edges, w, h, vp)`：**

隔离带检测完全基于边缘图中的纵向直线。不同于普通障碍物的轮廓检测，隔离带在图像中是纵向延伸数十到数百像素的连续直线段。

第一步，`cv2.HoughLinesP` 提取所有直线段。阈值参数来自 `config.HOUGH_*`——`HOUGH_THRESHOLD`（默认 50）控制需要多少边缘点投票才能算一条线，值越大越苛刻；`HOUGH_MIN_LINE_LEN`（默认 40 像素）控制最小线段长度；`HOUGH_MAX_LINE_GAP`（默认 30 像素）控制允许的最大断线连接距离。

第二步，角度筛选。每条线段的倾角必须在 `[LINE_ANGLE_MIN, LINE_ANGLE_MAX]`（约 70° 到 110°）之间。

第三步，消失点约束（可选）。如果标定器已收敛，计算每条纵向直线到消失点的距离。距离超出 `LINE_VP_DISTANCE_THRESHOLD`（默认 30 像素）的直线被过滤。这一步极为有效——地面上偶然的裂缝、光影、反光条的边缘虽然也接近纵向，但它们不会指向消失点。

第四步，聚类。取每条线段底部的 x 坐标（较大的 y 坐标对应的 x）。底部 x 坐标小于 `BARRIER_LEFT_MAX × w`（默认 ROI 宽度 25%）的归入左簇，大于 `BARRIER_RIGHT_MIN × w`（默认 70%）的归入右簇。每簇取中位数作为当帧检测结果。

**时序滤波。** 三个子检测的每帧结果分别入队。最终输出取队列中位数而非最新值。中值滤波对偶发的单帧误检（如某帧恰好有一道光影落在搜索范围内）有很好的抑制效果，同时不会引入类似滑动平均的滞后。窗口大小 10 帧在 30fps 下对应约 0.33 秒的延迟，对结构检测来说可接受——隔离沟和隔离带在正常情况下不会快速移动。

**可视化。** `_draw_boundaries()` 在 debug 图像上绘制：绿色实线 = 隔离沟位置，黄色实线 = 左右隔离带，蓝色虚线 = 车道中心线（隔离沟与左侧隔离带的中点，用于指示车辆的理想行驶轨迹），右下角标注置信度。

### 3.3 DebrisDetector — 碎石碎渣检测

**问题背景。** 碎石、水泥碎块、掉落的小工具与灰色水泥地面颜色完全相同。HSV 颜色分割无法将它们从地面中区分出来。但是，光滑平整的路面具有均匀的低纹理特征，碎石的堆积会破坏这种一致性——在边缘图中表现为局部区域边缘密度异常升高。

**算法选择。** 基于统计纹理分析的异常检测。不依赖颜色，不依赖形状模板，只依赖边缘密度的局部统计差异。方法简单（不需要训练），但对颜色不变（水泥地和灰色碎石都可以），计算量小。

**`detect(roi_gray, roi_edges, lane_boundary, calibrator)`：**

第一步，确定搜索范围。如果车道边界检测有效，搜索范围限定在 `left_barrier_px` 到 `ditch_px` 之间——只在自车车道内搜索碎石，忽略隔离沟对面和隔离带外侧的区域。如果边界检测无效，使用整个 ROI 宽度。

第二步，网格划分。将搜索范围内的边缘图划分为 `DEBRIS_GRID_CELL × DEBRIS_GRID_CELL`（默认 20×20 像素）的规则网格。如果列数或行数不足 2，返回空结果。

第三步，边缘密度计算。遍历每个网格单元，计算单元内边缘像素数占总像素数的比例。平滑路面单元密度低（0.01~0.05），碎石区域单元密度高（0.10~0.30）。结果存入 `density_map` 二维数组。

第四步，异常检测。对 `density_map` 的每个单元，取其 3×3 邻域（边界处截断），计算邻域的均值 μ 和标准差 σ。如果当前单元的密度 > μ + `DEBRIS_EDGE_DENSITY_STD_MULT`（默认 2.0）× σ，标记为异常。将标准差乘数调大可以减少误检，调小可以检测更细微的碎石。邻域大小至少需要 3 个单元才有统计意义。

第五步，合并异常。在二值异常图上用 `cv2.connectedComponents` 做 8-邻接连通域分析，将相邻异常单元合并为碎石斑块。每个斑块至少包含 2 个异常单元。

第六步，尺寸换算。如果标定器已收敛，取斑块中心行的 `meters_at_row()` 获取水平每像素米数。斑块的近似物理尺寸 = 斑块面积的平方根 × 每像素米数 × 100。标定未收敛时尺寸为 0（仅做像素级检测不做分类）。

第七步，分类输出。尺寸小于 `DEBRIS_SMALL_CM`（默认 5cm）的忽略。5~15cm 的记录但不标记为大石块。超过 `DEBRIS_LARGE_CM`（默认 15cm）的标记 `has_large_debris = True`，由路径规划模块在构建占用栅格时考虑避让。搜索范围的边界在 debug 图像上以灰色竖线标注。

**局限与注意。** 此方法在路面本身纹理不均匀（如粗糙混凝土、有补丁的沥青）时可能产生大量误检。网格单元大小和标准差乘数需要根据实际路面纹理调参。另外，标定未收敛时无法正确分类碎石尺寸，只能做"有无"的二元判断。

### 3.4 PathPlanner — 车道内路径规划

**问题背景。** 当前决策器是逐帧反应式的：本帧中心被堵了就选左右更畅通的一侧绕行。单帧决策不评估未来几步的后果。当遇到连续交错分布的障碍物时——比如先左绕再右绕——反应式策略会在帧间来回切换方向，车辆蛇形摆动。更严重的情况是非凸障碍布局（如两个障碍物形成的门缝状间隙）：反应式策略可能判断一侧可通、进入后发现被第二个障碍挡住、倒车、再选另一侧、重复。需要一个在车道约束内评估多步路径、选全局最优的规划器。

**算法选择。** DWA（Dynamic Window Approach）是机器人局部路径规划的标准方法。在速度空间中采样候选（线速度, 角速度），前向模拟轨迹，评估每条轨迹的安全性、平滑性和进度，选最优。之所以不选全局规划（A*、RRT），是因为隧道内没有先验地图，且半幅路内的规划空间本来就很窄（可用宽度 1~2 米），DWA 的局部 5 米前视距离在这种场景下足够了。

我们对标准 DWA 做了简化：不采样连续的速度空间，而是采样离散的曲率（对应固定的转弯半径）。原因是低速施工车辆的线速度变化范围很小（10~30 km/h），主要决策维度是转向角而不是速度。

**初始化。** `_last_steering` 记录上一帧的转向角，用于平滑性评分。

**`plan(...)`：**

第一步，可通行性预检。从 `lane_boundary.lane_width_m` 获取米制车道宽度。如果车道宽度 < 车身宽度 + 2 × 安全余量，直接返回 `PathPlan(passable=False, blocked_reason="LANE_TOO_NARROW")`。车道宽度为零（标定未收敛或边界检测无效）时，跳过此检查，假设可通过。

第二步，生成候选曲率。调用 `_generate_curvatures()` 生成 `PATH_NUM_CURVATURES`（默认 7）个均匀分布的曲率值，从 `-PATH_MAX_CURVATURE` 到 `+PATH_MAX_CURVATURE`（默认 0.35/m）。正值表示右转，负值表示左转，0 表示直行。曲率 0.35/m 对应约 2.9 米转弯半径——对于一个轴距约 2 米的施工车来说合理。

第三步，构建占用区域。`_build_occupied_zones()` 将来自多个模块的障碍信息转换为地面坐标系中的一组矩形"禁入区"：车道左侧边界外（隔离带外侧）和右侧边界外（隔离沟外侧）各一个 0.3 米宽的禁入区（防止车身超出车道）；碎石检测中的大石块（超过 15cm）在车道中部的禁入区（半径 = 石块厘米尺寸 / 200）；自由空间中评分很低的中心区域（`center_free_score < 0.3`）对应的禁入区。所有禁入区以米为单位。

第四步，轨迹评估。对每条候选曲率，调用 `_evaluate_trajectory()` 模拟前向滚动。轨迹起点为车道中心（标定可用时）或半幅车道中部（标定不可用时），方向为正前方。模拟步长 `PATH_STEP_LENGTH_M`（默认 0.3 米），共 `PATH_LOOKAHEAD_M / PATH_STEP_LENGTH_M` 步（默认 5/0.3 ≈ 17 步）。每一步用自行车模型更新位置：

```text
如果曲率 ≈ 0（直行）:
    新位置: x += step × sin(θ), y += step × cos(θ)
如果曲率 ≠ 0（转弯）:
    转弯半径 R = 1 / 曲率
    角度增量 dθ = step / R
    新位置: x += R × (cos(θ) - cos(θ+dθ))
            y += R × (sin(θ+dθ) - sin(θ))
    新朝向: θ += dθ
```

每一步检查当前位置是否落在禁入区内（到矩形的最短距离 < `PATH_CLEARANCE_MIN_CM / 100` 米即判为碰撞，score = -1000）。同时跟踪全程的最小安全间距。

轨迹评分使用三因子加权和：

- 安全分（权重 `PATH_WEIGHT_CLEARANCE`，默认 0.5）= 全程最小间距（米）。间距越大安全性越高。
- 平滑分（权重 `PATH_WEIGHT_SMOOTHNESS`，默认 0.3）= `1.0 - |曲率| / PATH_MAX_CURVATURE`。曲率越大（转弯越急）平滑分越低。
- 前进分（权重 `PATH_WEIGHT_PROGRESS`，默认 0.2）= 轨迹总前进距离 / 前视距离。鼓励选择沿着车道方向前进的轨迹而非原地打转。

此外，轨迹终点如果落在车道外（x < 0 或 x > lane_width），扣除 500 分（但不判为碰撞——因为终点偏出一点可能只是角度没对准，不代表路径不可行）。

第五步，选最优。`best_score` 初始为负无穷。遍历七条轨迹，如果某条轨迹得分最高且 > -100（碰撞线），选为最优。如果所有轨迹都 < -100，输出 `passable=False`。

第六步，输出。将最优曲率通过 `_curvature_to_steering()` 映射为转向角（线性映射到 `-STEERING_LARGE ~ +STEERING_LARGE`）。根据最优得分通过 `_adapt_speed()` 映射为速度（得分高则保持默认速度，中等则降到 70%，低则降到慢速）。最后通过帧间限幅（`STEERING_MAX_RATE`，默认 0.3）平滑转向角变化。

---

## 第四章：配置系统

所有可调参数集中在 `config.py`，按功能分九组。每组以注释分隔。

参数的设计原则是：每个参数对应一个明确的物理含义或算法行为，修改参数即可改变系统行为而无需改动算法代码。参数命名使用大写蛇形命名。

现有参数组（Phase 1 未修改）：**摄像头**（设备索引、分辨率、帧率、letterbox 开关）、**ROI**（四个比例参数，决定分析区域大小）、**预处理**（高斯核、CLAHE 参数、Canny 双阈值、形态学参数、HSV 地面分割上下界）、**可行驶区域**（三区比例、边缘密度阈值、Hough 参数）、**障碍检测**（面积上下限、宽高比范围、危险区比例）、**决策**（目标 FPS、最大延迟、防抖帧数、速度/转向各级参数、tanh 参数、帧间限幅）、**调试**（可视化开关、视频/日志保存路径）、**断线重连**（退避参数）、**语音输入**（录音设备、临时文件路径）。

Phase 1 新增四组参数：**标定**（相机安装几何、消失点收敛参数）、**结构检测**（隔离沟搜索范围和阈值、隔离带聚类范围、霍夫线筛选角度、时序滤波窗口）、**碎石检测**（网格尺寸、异常判定倍数、分类尺寸门槛）、**路径规划**（候选曲率数量、前视距离、步长、安全间距、评分权重、最大曲率、内轮差补偿）。

关键参数的调参指南已在前面各模块的算法描述中随文说明。所有估算参数（相机高度、俯仰角、车身尺寸）在 config 中以注释标注了"⚠️ 待实测"。

---

## 第五章：可视化调试系统

`build_debug_display()` 是 Phase 1 扩展后的调试面板。

**上排（四点五英寸，180×320 像素每面板）：** 原始画面（标注 "Original"）、ROI 裁剪区域（标注 "ROI"）、Canny 边缘图（标注 "Edges"）、HSV 二值 mask（标注 "Binary Mask"）。

**下排（二点四英寸，240×320 像素每面板）：** 自由空间（含三区划分线和各区域评分，Phase 1 扩展为当车道边界有效时使用动态分区标注）、障碍检测（含边界框和危险等级标签）、车道边界（Phase 1 新增，含绿色隔离沟线、黄色隔离带线、蓝色车道中心线、右下置信度）、路径规划（Phase 1 新增，显示规划的轨迹和禁入区）、决策面板（含 FPS、延迟、指令、速度/转向、原因、障碍物阻挡状态、危险等级、自由空间评分、偏移量）。

任何面板对应的模块如果未启用或返回 None，该面板显示黑底。

---

## 第六章：当前状态与后续

代码全部实现并测试通过。空管道延迟约 5ms/帧。完整主循环在 test_video.mp4 上达到 163 FPS（远超 30fps 目标）。所有新增模块的参数默认可选，不提供时系统行为与修改前一致。

当前主要限制是缺少包含隔离沟和隔离带的真实隧道场景视频来验证结构检测模块。获取实车数据和隧道视频后，需要优先更新 config 中的相机几何参数、然后调参验证结构检测精度、最后在带障碍物的场景上测试路径规划。

Phase 2 将处理极端光照（大灯过曝检测、AE 切换处理、低照度降噪），接入 YOLOv8 + ByteTrack 做动态障碍物的检测与跟踪，将路径规划从简化 DWA 升级为标准 DWA。Phase 3 覆盖恶劣天气、传感器退化和 Fail-Safe 安全链路。

---

## 第七章：v2.1 生产级安全修复（2026-06-04）

v2.1 是一次专项安全修复版本。经千问(Qwen 3.7 Max)深度代码审查，发现 8 个会影响实车安全的软件缺陷。本章详述每个问题的根因和修复方案。

### 7.1 安全降级迟滞效应（safety_degrader.py）

**问题根因。** v1.0 的 `evaluate()` 方法在置信度恢复时瞬间将降级等级从 L3/L2 拉回 L0（正常，30km/h）。当传感器在阈值边缘震荡时（如置信度在 0.49↔0.51 间波动），车辆会在"急刹停车"和"全速冲刺"之间高频切换。

**修复方案。** 引入双阈值迟滞机制：

- **降级阈值**（低）：L1=0.50, L2=0.30, L3=0.10，低于此值立即降级
- **恢复阈值**（高）：L1=0.70, L2=0.50, L3=0.30，高于此值才开始计恢复帧
- **恢复确认帧数**：`recovery_frames=10`，需连续满足恢复阈值 10 帧才逐级恢复（L3→L2→L1→L0）
- 降级方向立即执行，恢复方向延迟确认——安全优先

新增参数：`l1_recovery_confidence=0.70`, `l2_recovery_confidence=0.50`, `l3_recovery_confidence=0.30`, `recovery_frames=10`。新增状态变量：`_recovery_frame_count`。

### 7.2 L3 超时计时器错误重置（safety_degrader.py）

**问题根因。** v1.0 中，进入 L2_CONSERVATIVE 或 L1_CAUTION 状态时，`_l3_entry_time` 被设为 None。如果系统在 L2 和 L3 之间震荡（如置信度在 0.08↔0.12 间波动），每次回到 L2 都会清零计时器，导致 L3 累计时间永远达不到 10 秒，L3→L4（物理抱死）的安全机制永远无法触发。

**修复方案。** `_l3_entry_time` 只在恢复到 L0_NORMAL 时才重置。其他降级路径（L1→L2→L3）保留原值。这保证了 L3 的累计时间统计是连续的。

### 7.3 阿克曼转向非线性映射（path_planner.py）

**问题根因。** v1.0 的 `_curvature_to_steering()` 使用线性映射：`steer = (curvature / max_curvature) × STEERING_LARGE`。真实的阿克曼转向几何中，曲率 κ 与转向角 δ 的关系为 `κ = tan(δ) / L`（L 为轴距）。线性映射在大转角时产生严重偏离，导致车辆实际轨迹与规划轨迹不一致。

**修复方案。** 改用阿克曼运动学反解：
```python
steer_rad = math.atan(curvature * wheelbase)
steer = steer_rad / max_steer_rad  # 归一化到 [-1, 1]
```
新增 config 参数：`STEERING_MAX_ANGLE_DEG = 30.0`（最大转向角，用于归一化分母）。

### 7.4 dt-based 帧间转向限幅（path_planner.py）

**问题根因。** v1.0 的帧间限幅 `delta = max(-STEERING_MAX_RATE, min(STEERING_MAX_RATE, delta))` 限制的是每帧变化量，与帧率无关。如果 FPS 从 30 掉到 10，实际转向角速度（度/秒）骤降 3 倍，转弯不及；反之间隔缩短导致机械冲击。

**修复方案。** 引入基于时间差 dt 的限幅：
```python
max_delta = STEERING_MAX_RATE_DEG_PER_SEC * (π / 180) * dt
```
新增 config 参数：`STEERING_MAX_RATE_DEG_PER_SEC = 60.0`（最大转向角速度，度/秒）。dt 通过 `time.time()` 计算，默认回退为 1/30s。

### 7.5 占用区域精细化投影（path_planner.py）

**问题根因。** v1.0 的 `_build_occupied_zones()` 中，只要 `center_free_score < 0.3` 就把车道 20%~80%、纵向 1~5m 的矩形区域全部标记为不可通行。轻度障碍（如 0.25 的评分）与完全阻塞（0 评分）被同等对待，直接破坏了"中心阻塞→侧向绕行"的设计初衷。

**修复方案。** 改为三级占用模型：
- 评分 = 0（完全阻塞）：标记完整区域
- 0 < 评分 < 0.3（部分阻塞）：标记缩小 50%~80% 的占用范围，留出绕行空间
- 评分 ≥ 0.3（正常）：不标记占用

### 7.6 碰撞检测向量化（path_planner.py）

**问题根因。** v1.0 的 `_evaluate_trajectory()` 使用 Python 嵌套 for 循环做碰撞检测：对每步轨迹的每个采样点遍历所有 occupied 矩形。当 occupied 较多时，O(N×M×K) 的纯 Python 循环耗时可能超过 100ms。

**修复方案。** 将 occupied 预转换为 numpy 数组，新增 `_dist_to_rects_vectorized()` 方法一次性计算点到所有矩形的距离：
```python
cx = np.maximum(rects[:, 0], np.minimum(x, rects[:, 1]))
cy = np.maximum(rects[:, 2], np.minimum(y, rects[:, 3]))
return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
```

### 7.7 隔离沟方向动态判断（decision_maker.py）

**问题根因。** v1.0 的 `_is_crossing_ditch()` 硬编码假设"隔离沟在车辆右侧"（`offset > 0` 且 `right_free_score` 高即为越沟）。在双向隧道中，如果小车在右半幅行驶，隔离沟实际上在左侧，该逻辑完全失效。

**修复方案。** 通过 `lane_boundary.ditch_px` 与 `left_barrier_px`、`right_barrier_px` 的相对位置动态判断隔离沟在左侧还是右侧：
- 沟在右：offset > 阈值×2 且 right_free_score 异常高 → 越沟
- 沟在左：offset < -阈值×2 且 left_free_score 异常高 → 越沟

### 7.8 时序滤波跳变抑制（lane_boundary_detector.py）

**问题根因。** v1.0 的时序中值滤波直接将每帧检测结果入队取中位数，不对新值做任何检验。暗色井盖、水渍等偶发干扰在连续数帧内被误检为隔离沟时，中值会瞬间跳变几十像素，需要多帧才能恢复。

**修复方案。** 新增 `_innovation_check()` 新息检验：历史至少 3 帧时，计算当前中位数，新值偏离中位数超过 `BOUNDARY_MAX_JUMP_PX`（默认 50 像素）则拒绝加入队列。这防止了野值污染滤波器，同时不影响正常渐变（如车辆缓慢偏移时检测结果的自然漂移）。

新增 config 参数：`BOUNDARY_MAX_JUMP_PX = 50`。

### 7.9 新增配置参数汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `STEERING_MAX_RATE_DEG_PER_SEC` | 60.0 | dt-based 转向角速度限制 |
| `STEERING_MAX_ANGLE_DEG` | 30.0 | 阿克曼非线性映射的最大转向角 |
| `BOUNDARY_MAX_JUMP_PX` | 50 | 时序滤波跳变抑制阈值 |

以及 safety_degrader 内部的迟滞恢复参数（l1/l2/l3_recovery_confidence、recovery_frames）。
