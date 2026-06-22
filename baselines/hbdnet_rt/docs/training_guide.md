# HBD-Net-RT v1.0 训练工程方案

## 0. 前置条件

- PyTorch >= 2.0, CUDA >= 11.8
- GPU 显存 >= 8 GB (batch_size=8, 640×384)
- 标注数据 >= 2000 张隧道关键帧
- SAM (Segment Anything Model) 用于粗标注加速

---

## 1. 数据采集与标注规范

### 1.1 采集要求

| 项目 | 要求 |
|------|------|
| 分辨率 | 1280×720 或更高, 竖屏 |
| 帧率 | 30 fps |
| 场景覆盖 | 正常光照(60%), 低照(15%), 过曝(10%), 出入口过渡(10%), 其他(5%) |
| 障碍物覆盖 | 每类目标至少 200 个实例 (工人/车辆/碎石/悬挂物) |
| 抽帧间隔 | >= 2 秒或场景变化 >30% (避免连续相似帧) |

### 1.2 文件命名与目录结构

```
datasets/tunnel/
├── images/
│   ├── train/
│   │   ├── scene01_00001.jpg
│   │   ├── scene01_00003.jpg   # 抽帧: 跳过了00002
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── annotations/
│   ├── train/
│   │   ├── scene01_00001.json  # COCO格式检测标注
│   │   ├── scene01_00001_ego_passable.png    # 单通道PNG, 0/255
│   │   ├── scene01_00001_hard_boundary.png   # 4通道PNG, 0/255每通道
│   │   └── ...
│   ├── val/
│   └── test/
└── splits/
    ├── train.txt    # 每行一个图片路径(相对images/)
    ├── val.txt
    ├── test.txt
    └── hard_samples.txt   # 困难样本索引(Stage 4使用)
```

### 1.3 标注规范

#### 检测标注 (COCO JSON格式)

```json
{
  "images": [{"id": 1, "file_name": "scene01_00001.jpg", "width": 1280, "height": 720}],
  "categories": [
    {"id": 0, "name": "construction_vehicle"},
    {"id": 1, "name": "worker"},
    {"id": 2, "name": "suspended_object"},
    {"id": 3, "name": "falling_debris"}
  ],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "bbox": [320, 180, 80, 200],        // [x, y, w, h]
      "area": 16000,
      "iscrowd": 0,
      "ignore": 0,                         // 0=正常, 1=ignore(不参与loss)
      "difficulty": "normal"               // normal / hard
    }
  ]
}
```

**标注规则**:
- 遮挡超过 50% 的目标: 标 `ignore=1`, 不参与 loss 但保留在数据集中
- 小于 16×16 像素的极小目标: 标 `ignore=1`
- 工人蹲姿: 归入 worker 类; 手持长工具的工人: 工具部分不单独标, 人体 bbox 扩展 20%
- 施工车辆: 挖掘机/铲车/搅拌车/卡车全部分并为一个类 construction_vehicle
- 悬挂物: 只标可能进入车道范围的 (在图像下半部)

#### 分割标注 (PNG mask)

**ego_passable_mask** (单通道, 0=不可通行, 255=可通行):
- 标注本车所在半幅车道内所有可通行的水泥/沥青路面
- 包含: 有碎石覆盖但仍可通过的路面, 浅水洼
- 不包含: 隔离沟, 隔离带, 隧道壁, 对向车道(隔离沟另一侧)
- 最关键的规则: **隔离沟对面即使地面完全平整, 也必须标为0**
- 标注工具: SAM自动分割 → 人工修正边界

**hard_boundary_mask** (4通道PNG, 每通道独立):
```
通道0: 中央隔离沟 (ditch)         → 255=是边界, 0=否
通道1: 左侧隔离带 (left_barrier)   → 255=是边界, 0=否
通道2: 右侧隔离带 (right_barrier)  → 255=是边界, 0=否
通道3: 隧道壁 (tunnel_wall)       → 255=是边界, 0=否
```
- 标注时沿结构内侧边缘描边
- 边缘 3 像素范围内也标为边界 (模糊过渡区)
- 同一像素可属于多个类别 (如隔离沟紧挨隧道壁)

**edge_mask** (自动生成, 不人工标注):
```python
# 从 hard_boundary_mask 自动生成
edge = cv2.dilate(boundary, (3,3)) - cv2.erode(boundary, (3,3))
edge = (edge > 0).astype(np.uint8) * 255
```

### 1.4 标注加速流程

```
1. 用 SAM (segment-anything) 对每张图做粗分割
   → 输出: all_masks, scores
2. 人工选择对应类别的 mask
   → ego_passable: 选覆盖半幅路面的 mask
   → hard_boundary: 选覆盖隔离沟/隔离带的 mask
3. 人工修正边界 (耗时最大的步骤)
   → 重点修正: 隔离沟边缘, 隔离带与路面的分界, 碎石区域
4. 标注检测框 (用 LabelImg 或 CVAT)
5. 导出 COCO JSON + PNG masks
6. 自动生成 edge masks + hard_samples.txt
```

---

## 2. 数据加载

### 2.1 Transform Pipeline

```python
# 训练时的数据增强
train_transform = Compose([
    # 几何增强
    RandomHorizontalFlip(p=0.5),             # 左右翻转(检测框和mask同步翻转)
    RandomRotation(degrees=5),               # ±5°旋转
    RandomScale(scale_range=(0.8, 1.2)),     # 缩放

    # 光度增强
    RandomBrightnessContrast(
        brightness_limit=0.3,                # 亮度 ±30%
        contrast_limit=0.2                   # 对比度 ±20%
    ),
    RandomGamma(gamma_limit=(0.7, 1.5)),     # Gamma校正

    # 隧道特化增强
    RandomOverexposure(p=0.15),              # 模拟大灯过曝: 随机区域V+50
    RandomLowLight(p=0.15),                  # 模拟低照: V×0.3 + 高ISO噪声
    RandomFog(p=0.1),                        # 模拟光幕散射: 全局对比度×0.5

    # 噪声与遮挡
    GaussNoise(std_range=(5, 15), p=0.2),   # 高斯噪声
    RandomErasing(scale=(0.02,0.1), p=0.1), # 模拟局部遮挡

    # 归一化
    Normalize(mean=[0.485,0.456,0.406],     # ImageNet统计
              std=[0.229,0.224,0.225]),
    Resize(size=(384, 640)),                 # 统一尺寸
])

# 验证时的变换 (无增强)
val_transform = Compose([
    Normalize(mean=[0.485,0.456,0.406],
              std=[0.229,0.224,0.225]),
    Resize(size=(384, 640)),
])
```

### 2.2 DataLoader 配置

```python
train_loader = DataLoader(
    TunnelDataset(split='train', transform=train_transform),
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn,       # 处理不同尺寸的bbox
)

val_loader = DataLoader(
    TunnelDataset(split='val', transform=val_transform),
    batch_size=8,
    shuffle=False,
    num_workers=2,
)
```

### 2.3 困难样本重采样 (Stage 4)

```python
class HardSampleSampler(Sampler):
    """
    Stage 4专用采样器。
    每个batch: 50%样本从hard_samples索引中随机抽取,
              50%从全数据集随机抽取。

    初始化时从文件读取困难样本索引列表。
    """
    def __init__(self, dataset, hard_indices_file, batch_size):
        self.dataset = dataset
        self.hard_indices = self._load_hard_indices(hard_indices_file)
        self.easy_indices = list(set(range(len(dataset))) - set(self.hard_indices))
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)):
            n_hard = self.batch_size // 2
            n_easy = self.batch_size - n_hard
            batch = (
                np.random.choice(self.hard_indices, n_hard, replace=True).tolist() +
                np.random.choice(self.easy_indices, n_easy, replace=True).tolist()
            )
            np.random.shuffle(batch)
            yield batch
```

---

## 3. 损失函数实现

### 3.1 Detection Loss: Focal Loss

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    FL(pt) = -α_t * (1 - pt)^γ * log(pt)

    α=0.25: 平衡正负样本权重
    γ=2.0:  降低易分样本的权重, 让模型关注困难样本
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: [N, C] logits
        target: [N] class indices (0=C-1为前景, C=背景)
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
```

### 3.2 Segmentation Loss: Tversky Loss

```python
class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation.
    TL = 1 - (TP + smooth) / (TP + α*FP + β*FN + smooth)

    用于 ego_passable (α=0.7, β=0.3): 偏向召回
    用于 hard_boundary (α=0.5, β=0.5): 等同精度和召回
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W] logits
        target: [B, 1, H, W] 0/1
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1 - tversky
```

### 3.3 Boundary Loss: Dice + CrossEntropy

```python
class DiceCELoss(nn.Module):
    """
    Dice Loss + CrossEntropy 组合。
    用于 hard_boundary 分割 (多类)。
    Dice 关注小目标覆盖, CE 保证全局语义。
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        pred: [B, C, H, W] logits
        target: [B, H, W] class indices (0,1,2,3)
        """
        ce_loss = self.ce(pred, target)

        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        # 对每个类别计算 Dice
        dice = 0
        for c in range(pred.shape[1]):
            p_c = pred_soft[:, c].contiguous().view(-1)
            t_c = target_onehot[:, c].contiguous().view(-1)
            intersection = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum()
            dice += (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice / pred.shape[1]

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
```

### 3.4 总 Loss 组装

```python
class HBDNetRTLoss(nn.Module):
    """多任务总损失。"""
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = {
                'detection': 1.0,
                'passable': 1.0,
                'boundary': 1.2,
                'edge': 0.8,
                'risk': 0.8,
            }
        self.weights = weights
        self.det_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.passable_loss_fn = TverskyLoss(alpha=0.7, beta=0.3)
        self.boundary_loss_fn = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        self.edge_loss_fn = EdgeWeightedBCE(edge_weight=3.0, edge_radius=3)
        self.risk_loss_fn = nn.MSELoss()    # Surface risk

    def forward(self, outputs, targets):
        L_det = self.det_loss_fn(
            outputs['detection_logits'], targets['detection_labels'])

        L_passable = self.passable_loss_fn(
            outputs['ego_passable_mask'], targets['ego_passable_mask'])

        L_boundary = self.boundary_loss_fn(
            outputs['hard_boundary_mask'], targets['hard_boundary_mask'])

        L_edge = self.edge_loss_fn(
            outputs['hard_boundary_edge'], targets['hard_boundary_edge'])

        L_risk = self.risk_loss_fn(
            outputs['surface_risk_map'], targets['surface_risk_map'])

        total = (self.weights['detection'] * L_det +
                 self.weights['passable'] * L_passable +
                 self.weights['boundary'] * L_boundary +
                 self.weights['edge'] * L_edge +
                 self.weights['risk'] * L_risk)

        return total, {
            'detection': L_det.item(),
            'passable': L_passable.item(),
            'boundary': L_boundary.item(),
            'edge': L_edge.item(),
            'risk': L_risk.item(),
            'total': total.item(),
        }
```

---

## 4. 训练循环

### 4.1 优化器与调度器

```python
# Stage 1: 只训练Head
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Stage 2: 降低学习率
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-4

# Stage 3: 进一步降低
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-5
```

### 4.2 冻结/解冻策略

```python
def freeze_backbone(model):
    """Stage 1: 冻结 Backbone"""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.neck.parameters():
        param.requires_grad = True

def unfreeze_neck_and_deep_backbone(model, unfreeze_stages=2):
    """Stage 2: 解冻 Neck + Backbone 后N个stage"""
    for param in model.neck.parameters():
        param.requires_grad = True
    stages = list(model.backbone.stages)
    for stage in stages[-unfreeze_stages:]:
        for param in stage.parameters():
            param.requires_grad = True

def unfreeze_all(model):
    """Stage 3: 全模型解冻"""
    for param in model.parameters():
        param.requires_grad = True
```

### 4.3 训练主循环

```python
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    epoch_losses = defaultdict(float)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in targets.items()}

        with torch.cuda.amp.autocast():                    # AMP混合精度
            outputs = model(images)
            total_loss, loss_dict = criterion(outputs, targets)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        for k, v in loss_dict.items():
            epoch_losses[k] += v

        if batch_idx % 50 == 0:
            log_training_step(batch_idx, loss_dict)

    return {k: v / len(loader) for k, v in epoch_losses.items()}
```

### 4.4 Checkpoint 管理

```python
checkpoint = {
    'epoch': epoch,
    'stage': current_stage,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_metrics': best_metrics,
    'config': config_dict,
}
torch.save(checkpoint, f'checkpoints/stage{stage}_epoch{epoch}.pth')

# 保留最佳模型 (按 val boundary IoU)
if val_metrics['boundary_iou'] > best_metrics['boundary_iou']:
    torch.save(checkpoint, 'checkpoints/best_model.pth')
    best_metrics = val_metrics
```

---

## 5. 评估指标

```python
def compute_metrics(outputs, targets):
    metrics = {}

    # Detection
    metrics['detection_mAP'] = compute_map(
        outputs['detections'], targets['detection_annotations'], iou_thresh=0.5)
    metrics['detection_recall_worker'] = compute_recall(
        outputs['detections'], targets['detection_annotations'], class_id=1)

    # Segmentation
    metrics['passable_mIoU'] = compute_miou(
        outputs['ego_passable_mask'], targets['ego_passable_mask'])
    metrics['boundary_IoU_ditch'] = compute_iou(
        outputs['hard_boundary_mask'][:, 0], targets['hard_boundary_mask'][:, 0])
    metrics['boundary_IoU_mean'] = compute_iou_multiclass(
        outputs['hard_boundary_mask'], targets['hard_boundary_mask'])

    # Surface Risk
    metrics['risk_mae'] = F.l1_loss(
        outputs['surface_risk_map'], targets['surface_risk_map']).item()

    return metrics
```

**目标指标 (训练收敛参考值)**:
| 指标 | Stage 1 后 | Stage 2 后 | Stage 3 后 | 最终目标 |
|------|-----------|-----------|-----------|---------|
| detection mAP@0.5 | ≥ 50% | ≥ 65% | ≥ 75% | ≥ 75% |
| worker recall | ≥ 70% | ≥ 85% | ≥ 90% | ≥ 90% |
| passable mIoU | ≥ 75% | ≥ 82% | ≥ 88% | ≥ 88% |
| boundary IoU (ditch) | ≥ 50% | ≥ 62% | ≥ 70% | ≥ 70% |
| risk MAE | - | - | ≤ 0.15 | ≤ 0.10 |

---

## 6. 训练执行命令

```bash
# Stage 1: 冻结Backbone, 训练Head
python train.py --stage 1 --epochs 50 --lr 1e-3 \
    --freeze-backbone --batch-size 8 \
    --data-root datasets/tunnel

# Stage 2: 解冻Neck+后层
python train.py --stage 2 --epochs 100 --lr 1e-4 \
    --unfreeze-stages 2 --batch-size 8 \
    --resume checkpoints/stage1_epoch50.pth

# Stage 3: 全模型微调
python train.py --stage 3 --epochs 50 --lr 1e-5 \
    --unfreeze-all --batch-size 8 \
    --resume checkpoints/stage2_epoch100.pth

# Stage 4: 困难样本重采样
python train.py --stage 4 --epochs 50 --lr 1e-5 \
    --hard-sample-mode --hard-ratio 0.5 \
    --resume checkpoints/stage3_epoch50.pth
```

---

## 7. 部署导出

```bash
# 1. PyTorch → ONNX
python scripts/export_onnx.py \
    --weights checkpoints/best_model.pth \
    --input-size 384 640 \
    --output model.onnx

# 2. 验证 ONNX 精度 (对比 PyTorch 输出)
python scripts/verify_onnx.py \
    --pytorch checkpoints/best_model.pth \
    --onnx model.onnx \
    --tolerance 1e-3

# 3. ONNX → TensorRT FP16 (Jetson / GPU)
trtexec --onnx=model.onnx \
    --fp16 \
    --saveEngine=model_fp16.engine \
    --verbose

# 4. Benchmark
python scripts/benchmark_latency.py \
    --engine model_fp16.engine \
    --frames 1000 --warmup 100
```

---

## 8. 验收清单

训练完成后的验证步骤:

- [ ] 检测 mAP@0.5 ≥ 75%, worker recall ≥ 90%
- [ ] passable mIoU ≥ 88%, boundary IoU ≥ 70%
- [ ] TensorRT FP16 推理延迟 < 15ms
- [ ] 8 个决策场景全部通过 (test_scenarios.py)
- [ ] 正常光照直行: 可行驶区域正确, 隔离沟不漏检
- [ ] 靠近隔离沟: DWA 轨迹不跨越
- [ ] 近处工人(2m): 触发 STOP
- [ ] 过曝/低照: 触发降级, 不冒进
- [ ] 连续运行30分钟: 无内存泄漏, 无延迟漂移
