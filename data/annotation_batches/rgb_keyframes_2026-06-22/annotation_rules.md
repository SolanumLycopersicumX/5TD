# RGB Keyframe Annotation Rules

本批次用于 RGB 纯视觉路线的第一阶段训练数据：先学稳定的可通行区域和不可跨越硬边界；当前视频基本没有障碍物，目标框可以为空。

## 文件位置

- 图片目录：`images/`
- 固定标签：`labels.txt`
- 标注软件：Labelme

启动命令：

```bash
cd /home/tomato/5TD
labelme data/annotation_batches/rgb_keyframes_2026-06-22/images \
  --labels data/annotation_batches/rgb_keyframes_2026-06-22/labels.txt \
  --nodata
```

标注完成后，把每张图对应的 `.json` 保存到图片旁边。

## 必须标注

| 标注内容 | Labelme label | 形状 | 说明 |
|---|---|---|---|
| 本车可通行地面 | `ego_passable` | polygon | 只标本车这一侧能安全行驶的地面。 |
| 深沟、排水沟、中央沟 | `ditch` | polygon | 标出车辆绝对不能跨越的沟、槽或沟边区域。 |
| 左侧不可跨越边界 | `left_barrier` | polygon | 左侧路缘、隔离带、护栏、墙根障碍等。 |
| 右侧不可跨越边界 | `right_barrier` | polygon | 右侧轨道边、沟边、路缘、隔离带等。 |
| 隧道墙体或墙根不可通行区域 | `tunnel_wall` | polygon | 墙体、墙根、明显不可通行的贴墙区域。 |

## 后续有障碍物时再标

| 标注内容 | Labelme label | 形状 | 说明 |
|---|---|---|---|
| 人 | `worker` | rectangle | 包住完整人体；蹲姿也标为 `worker`。 |
| 工程车、卡车、铲车等 | `construction_vehicle` | rectangle | 同类工程车辆都用这个标签。 |
| 可能进入车体通行空间的悬挂物 | `suspended_object` | rectangle | 只标可能影响车辆通过的悬挂物。 |
| 碎石、工具、箱体、线缆、电机等障碍物 | `debris` | rectangle | 当前无法细分的小型或未知障碍统一用这个标签。 |

当前无障碍帧不要为了凑标签画框。没有目标框就是正确的负样本。

## 关键规则

1. `ego_passable` 只标本车所在半幅的可行驶地面。
2. 深沟对面即使看起来是平地，也不要标成 `ego_passable`。
3. 沟边、隔离带、轨道边、墙根等不能跨越的位置，优先标成对应 hard-boundary 标签。
4. 如果边界是一条窄线，polygon 可以沿边界画成一条窄带，不需要精确到单像素。
5. 反光、水渍、阴影如果仍然能安全通行，可以包含在 `ego_passable` 中；如果会遮挡沟边判断，额外保守地把不可确认区域排除。
6. 遮挡严重、看不清类别的障碍物，后续采集时先用 `debris`。
7. 不要手工标 `edge_mask` 或 `surface_risk`；`edge_mask` 后续会从 hard-boundary polygon 自动生成，`surface_risk` 第二阶段再补。

## 每张图的最低要求

当前这批无障碍视频，每张图至少检查并尽量标出：

- 一个 `ego_passable` polygon。
- 可见的 `ditch`、`left_barrier`、`right_barrier`、`tunnel_wall` polygon。
- 没有障碍物时，不画任何 rectangle。

如果某张图某类边界完全不可见，就不要凭空补画。
