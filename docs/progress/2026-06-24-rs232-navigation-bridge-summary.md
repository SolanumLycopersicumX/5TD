# RS232 导航桥接方案总结

日期：2026-06-24

这份总结用于把当前项目状态、RS232 驱动选择、旧代码处理方式和下一步实施计划上传给 ChatGPT 或交给其他人继续分析。

## 1. 当前项目状态

项目目标是让隧道小车根据视觉识别结果规划可行驶轨迹，并最终输出给底盘驱动。

目前最可用的感知路线不是早期 OpenCV 方法，也不是 SAM 标注方案，而是当前已经训练出的分阶段语义分割方案：

- 主模型识别 `ego_passable` 和 `ditch`。
- 辅助模型识别 `left_barrier` 和 `tunnel_wall`。
- 融合逻辑生成最终的 `safe_passable`。

当前建议使用的模型和脚本：

- 主模型：`runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt`
- 辅助模型：`runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt`
- 融合脚本：`tools/passable_segmentation/visualize_fused_passable_boundary.py`

当前融合规则：

- `ditch` 优先级最高。
- `tunnel_wall` 永远不可通行。
- `left_barrier` 只作为边界提示，不直接作为排水沟或不可通行区域。
- `safe_passable = passable - ditch - tunnel_wall`。

最近一次有效标注集上的指标：

- `safe_passable_iou`: 0.9605
- `ditch_iou`: 0.4699
- `left_barrier_iou`: 0.4976
- `tunnel_wall_iou`: 0.7978

结论：当前模型已经可以用于第一版离线轨迹规划实验，但数据仍然偏少，尤其是右侧排水沟的识别还需要继续补标和训练。

## 2. 驱动选择

当前决定使用 RS232 / Modbus RTU 作为底盘控制接口。

用户提供的 `1.zip` 中包含三个文件：

- `driver_controller.py`：RS232 / Modbus RTU 车辆控制代码，是本次驱动对接的主要参考。
- `can_vehicle_gui.py`：CAN 控制 GUI，本路线暂不采用。
- `IO_controller.py`：IO 控制，不是运动控制接口。

`driver_controller.py` 中的关键协议：

- 串口：115200 baud, 8N1。
- Modbus 节点地址：`0x06`。
- 线速度寄存器：`1040`，单位 `0.001 m/s`。
- 角速度寄存器：`1041`，单位 `0.001 rad/s`。
- 功能控制寄存器：`1045`，包含急停位。
- 驱动使能寄存器：`1049`，`1` 表示使能，`2` 表示失能。

## 3. 旧代码是否继续使用

结论：旧代码不应作为新系统的运行时依赖。

原因：

- 旧 baseline 把感知、规划和控制混在一起，不适合继续扩展。
- 旧代码中有早期 OpenCV / 障碍物避障路线的历史假设。
- 旧代码中的角速度方向注释与新 `driver_controller.py` 不一致。

旧代码只能作为参考，例如：

- 候选轨迹评分思路。
- 保守停车逻辑。
- RS232 寄存器写入方式对照。

新工程代码建议统一放在 `src/tunnel_nav` 下。

## 4. 当前最大安全风险

当前最重要的安全风险是角速度正负方向还不能完全确认。

冲突如下：

- 新 `driver_controller.py` 写的是：角速度为正时左转 / 逆时针。
- 旧 baseline 注释写的是：左负右正。

因此下一步必须：

- 增加 `angular_sign` 配置项。
- 默认只 dry-run，不真实打开串口。
- 实车运动前必须低速验证角速度方向。

## 5. 下一步实施计划

下一步不应直接让车运动，而是先实现离线导航桥接。

计划分为六步：

1. 添加运动数据结构：
   - `MotionCommand`
   - `NavigationConfig`
   - `MaskBundle`
   - `PathCandidate`

2. 添加 mask 轨迹规划和安全过滤：
   - 输入 `safe_passable`、`ditch`、`tunnel_wall`、`left_barrier`。
   - 只保留与图像底部连通的可行进区域。
   - 从图像底部中心生成若干候选轨迹。
   - 穿过 `ditch` 或 `tunnel_wall` 的轨迹必须拒绝。
   - 输出物理单位的 `linear_mps` 和 `angular_radps`，不再使用归一化 steering 作为最终控制接口。

3. 添加融合 mask 导出：
   - 导出 `safe_passable`、`ditch`、`left_barrier`、`tunnel_wall` 四类 PNG mask。
   - 保留当前已有可视化 overlay 行为。

4. 添加离线导航 CLI：
   - 输入图像文件夹和融合 mask 文件夹。
   - 输出每帧的 command JSON。
   - 输出轨迹可视化 overlay。
   - 不访问串口。

5. 添加 RS232 dry-run adapter：
   - 将 `linear_mps` 和 `angular_radps` 转成 Modbus 寄存器值。
   - 执行速度限幅。
   - 支持 `angular_sign` 方向翻转。
   - 默认只返回或打印将要写入的寄存器，不打开串口。

6. 添加配置和最终验证：
   - 更新 `configs/robot/vehicle.yaml`。
   - 新增导航桥接测试。
   - 运行已有分割工具测试。
   - 运行语法检查。

## 6. 计划新增或修改的文件

- `src/tunnel_nav/__init__.py`
- `src/tunnel_nav/motion.py`
- `src/tunnel_nav/mask_planner.py`
- `src/tunnel_nav/rs232.py`
- `tools/navigation_bridge/run_offline_navigation_bridge.py`
- `tools/passable_segmentation/visualize_fused_passable_boundary.py`
- `tests/test_navigation_bridge.py`
- `configs/robot/vehicle.yaml`
- `docs/progress/LOG.md`

详细实施计划已记录在：

- `docs/superpowers/plans/2026-06-24-rs232-navigation-bridge.md`

## 7. 第一版验收标准

第一版完成后应满足：

- 能从保存好的融合 mask 生成运动命令 JSON。
- 能生成轨迹可视化 overlay，便于人工检查。
- 不安全输入必须输出停车命令，并给出明确 reason。
- RS232 寄存器转换可以在无硬件情况下测试。
- 默认不打开串口。
- 新运行路径不 import 旧 baseline。
- 角速度方向可配置。
- 新测试和已有分割测试通过。

## 8. 推荐下一步

建议先实现离线版本，不要立即连接实车。

实车 RS232 写入前至少需要确认：

- 离线轨迹 overlay 看起来合理。
- 不安全区域能稳定触发停车。
- dry-run 输出的寄存器值正确。
- 急停和零速度写入逻辑明确。
- 角速度正负方向通过低速实车测试确认。
