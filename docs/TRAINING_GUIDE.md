# BraTS2021 脑肿瘤分割训练指南

本指南详细介绍如何使用 U-Net 2D 模型训练 BraTS2021 脑肿瘤分割任务。

## 目录

- [快速开始](#快速开始)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [训练配置](#训练配置)
- [训练命令](#训练命令)
- [高级功能](#高级功能)
- [训练监控](#训练监控)
- [常见问题](#常见问题)

---

## 快速开始

最简单的训练命令：

```bash
python train.py --data_dir ./data_png
```

这将使用默认配置开始训练，包括：
- 30 个训练轮次
- 批次大小为 8
- 学习率 1e-4
- 80% 训练集 / 20% 验证集划分

---

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch
- torchvision
- tqdm
- numpy
- Pillow

### 2. 检查 GPU 可用性

训练脚本会自动检测并使用 GPU（如果可用）。启动训练时会显示：

```
使用设备: cuda
GPU 型号: NVIDIA GeForce RTX 3090
GPU 显存: 24.00 GB
```

---

## 数据准备

### 数据集结构

确保数据集按以下结构组织：

```
data_png/
├── imgs/
│   ├── BraTS2021_00000_slice_050.png
│   ├── BraTS2021_00000_slice_051.png
│   └── ...
└── masks/
    ├── BraTS2021_00000_slice_050.png
    ├── BraTS2021_00000_slice_051.png
    └── ...
```

### 数据格式

- **输入图像**：4 通道 PNG 图像（T1, T1ce, T2, FLAIR）
- **标注掩码**：单通道 PNG 图像，像素值为 {0, 1, 2, 3}
  - 0: 背景
  - 1: 坏死和非增强肿瘤核心（NCR/NET）
  - 2: 水肿区域（ED）
  - 3: 增强肿瘤（ET）

### 数据转换

如果你有原始 BraTS2021 NIfTI 格式数据，可以使用转换脚本：

```bash
python scripts/convert_brats_to_png.py
```

---

## 训练配置

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data_png` | 数据集根目录 |
| `--epochs` | `30` | 训练轮数 |
| `--batch_size` | `8` | 批次大小 |
| `--lr` | `1e-4` | 初始学习率 |
| `--train_ratio` | `0.8` | 训练集比例 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--in_channels` | `4` | 输入通道数（4 种 MRI 模态）|
| `--num_classes` | `4` | 输出类别数（背景 + 3 类肿瘤）|
| `--base_channels` | `64` | U-Net 基础通道数 |
| `--bilinear` | `False` | 使用双线性插值上采样 |

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weight_decay` | `1e-5` | 权重衰减（L2 正则化）|
| `--num_workers` | `4` | 数据加载线程数 |

### 保存和日志

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_dir` | `./checkpoints` | 模型保存目录 |
| `--log_interval` | `10` | 日志打印间隔（批次数）|
| `--save_best_only` | `False` | 只保存最佳模型 |
| `--validate_every` | `1` | 每隔多少轮进行一次验证 |

---

## 训练命令

### 1. 基础训练

使用默认配置：

```bash
python train.py
```

### 2. 自定义数据路径和训练轮数

```bash
python train.py --data_dir ./data_png --epochs 50 --batch_size 16
```

### 3. 使用类别权重处理类别不平衡

```bash
python train.py --use_class_weights
```

脚本会自动计算每个类别的权重，给予少数类更高的权重。

### 4. 添加 Dice Loss

结合 CrossEntropyLoss 和 Dice Loss：

```bash
python train.py --dice_weight 0.5
```

总损失 = CrossEntropyLoss + 0.5 × Dice Loss

### 5. 启用学习率调度器

当验证集 Dice Score 停止改善时自动降低学习率：

```bash
python train.py --use_scheduler --scheduler_patience 5 --scheduler_factor 0.5
```

- `scheduler_patience`: 等待 5 轮无改善后降低学习率
- `scheduler_factor`: 学习率衰减因子（新学习率 = 旧学习率 × 0.5）

### 6. 启用早停机制

防止过拟合，当验证集性能不再提升时提前停止训练：

```bash
python train.py --early_stopping --patience 10
```

如果验证集 Dice Score 连续 10 轮未改善，训练将自动停止。

### 7. 恢复训练

从检查点继续训练：

```bash
python train.py --resume ./checkpoints/best_model.pth
```

### 8. 完整训练示例

结合多个高级功能的完整命令：

```bash
python train.py \
    --data_dir ./data_png \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_class_weights \
    --dice_weight 0.3 \
    --use_scheduler \
    --scheduler_patience 5 \
    --early_stopping \
    --patience 15 \
    --save_dir ./checkpoints \
    --num_workers 8 \
    --seed 42
```

---

## 高级功能

### 1. 类别权重（Class Weights）

**用途**：处理类别不平衡问题

BraTS 数据集中，背景像素远多于肿瘤像素。使用类别权重可以让模型更关注少数类。

```bash
python train.py --use_class_weights
```

训练时会显示计算的权重：

```
类别权重: [0.25, 2.5, 2.0, 3.0]
```

### 2. Dice Loss

**用途**：直接优化 Dice Score 指标

Dice Loss 与 Dice Score 直接相关，可以提升分割性能。

```bash
python train.py --dice_weight 0.5
```

推荐权重范围：0.3 - 0.7

### 3. 学习率调度

**用途**：动态调整学习率，提升收敛性能

使用 ReduceLROnPlateau 策略，当验证集性能停滞时降低学习率。

```bash
python train.py --use_scheduler --scheduler_patience 5 --scheduler_factor 0.5
```

### 4. 早停（Early Stopping）

**用途**：防止过拟合，节省训练时间

当验证集性能不再提升时自动停止训练。

```bash
python train.py --early_stopping --patience 10
```

### 5. 模型保存策略

**只保存最佳模型**：

```bash
python train.py --save_best_only
```

**同时保存最佳和最新模型**（默认）：

```bash
python train.py
```

保存的文件：
- `best_model.pth`: 验证集 Dice Score 最高的模型
- `latest_model.pth`: 最新一轮的模型
- `train_config.json`: 训练配置
- `train_history.json`: 训练历史记录

### 6. 随机种子

**用途**：确保实验可重复

```bash
python train.py --seed 42
```

---

## 训练监控

### 1. 实时进度条

训练过程中会显示实时进度：

```
Epoch 10/30 [训练]: 100%|████████| 125/125 [02:15<00:00, loss=0.3245, dice=0.7856, lr=0.000100]
Epoch 10/30 [验证]: 100%|████████| 32/32 [00:28<00:00, loss=0.2987, dice=0.8123]
```

### 2. Epoch 总结

每个 epoch 结束后会显示统计信息：

```
============================================================
Epoch 10 总结:
  训练损失: 0.3245 | 训练 Dice: 0.7856
  验证损失: 0.2987 | 验证 Dice: 0.8123
  学习率: 0.000100
============================================================
🎉 新的最佳模型！Dice Score: 0.8123
```

### 3. 训练历史

训练完成后，所有指标会保存到 `train_history.json`：

```json
{
    "train_loss": [0.5234, 0.4123, 0.3456, ...],
    "train_dice": [0.6543, 0.7234, 0.7856, ...],
    "val_loss": [0.4987, 0.3876, 0.3123, ...],
    "val_dice": [0.6789, 0.7456, 0.8123, ...],
    "lr": [0.0001, 0.0001, 0.00005, ...]
}
```

### 4. 模型信息

训练开始时会显示模型详细信息：

```
模型: U-Net 2D
  输入通道: 4
  输出类别: 4
  基础通道: 64
  上采样方式: 转置卷积
  总参数量: 31,042,692
  可训练参数: 31,042,692
```

---

## 常见问题

### Q1: 显存不足（CUDA out of memory）

**解决方案**：

1. 减小批次大小：
   ```bash
   python train.py --batch_size 4
   ```

2. 减小模型通道数：
   ```bash
   python train.py --base_channels 32
   ```

3. 使用双线性插值上采样（减少参数量）：
   ```bash
   python train.py --bilinear
   ```

### Q2: 训练速度慢

**解决方案**：

1. 增加数据加载线程数：
   ```bash
   python train.py --num_workers 8
   ```

2. 使用更大的批次大小（如果显存允许）：
   ```bash
   python train.py --batch_size 16
   ```

3. 减少验证频率：
   ```bash
   python train.py --validate_every 2
   ```

### Q3: 模型不收敛或 Dice Score 很低

**解决方案**：

1. 使用类别权重：
   ```bash
   python train.py --use_class_weights
   ```

2. 添加 Dice Loss：
   ```bash
   python train.py --dice_weight 0.5
   ```

3. 调整学习率：
   ```bash
   python train.py --lr 5e-4
   ```

4. 增加训练轮数：
   ```bash
   python train.py --epochs 100
   ```

### Q4: 如何恢复中断的训练？

使用 `--resume` 参数：

```bash
python train.py --resume ./checkpoints/latest_model.pth
```

### Q5: 如何查看训练配置？

训练配置会自动保存到 `checkpoints/train_config.json`：

```bash
cat checkpoints/train_config.json
```

### Q6: Dice Score 的含义是什么？

Dice Score 是医学图像分割中常用的评估指标，范围 [0, 1]：

- **0.0**: 完全不重叠（最差）
- **0.5**: 中等重叠
- **0.8+**: 良好重叠
- **1.0**: 完全重叠（完美）

本项目计算 3 类肿瘤区域（NCR/NET, ED, ET）的平均 Dice Score，忽略背景。

---

## 训练流程总结

1. **数据准备** → 确保数据集格式正确
2. **环境检查** → 安装依赖，检查 GPU
3. **配置参数** → 根据需求调整训练参数
4. **开始训练** → 运行 `python train.py` 命令
5. **监控进度** → 观察损失和 Dice Score 变化
6. **保存模型** → 最佳模型自动保存到 `checkpoints/`
7. **评估测试** → 使用 `scripts/evaluate.py` 评估模型性能

---

## 推荐训练配置

### 配置 1：快速测试（适合调试）

```bash
python train.py --epochs 10 --batch_size 4
```

### 配置 2：标准训练（推荐）

```bash
python train.py \
    --epochs 50 \
    --batch_size 8 \
    --use_class_weights \
    --dice_weight 0.3 \
    --use_scheduler \
    --early_stopping \
    --patience 10
```

### 配置 3：高性能训练（大显存 GPU）

```bash
python train.py \
    --epochs 100 \
    --batch_size 16 \
    --base_channels 64 \
    --use_class_weights \
    --dice_weight 0.5 \
    --use_scheduler \
    --scheduler_patience 5 \
    --early_stopping \
    --patience 15 \
    --num_workers 8
```

### 配置 4：低显存训练（小 GPU）

```bash
python train.py \
    --epochs 50 \
    --batch_size 2 \
    --base_channels 32 \
    --bilinear \
    --use_class_weights \
    --num_workers 2
```

---

## 下一步

训练完成后，你可以：

1. **评估模型**：使用 `scripts/evaluate.py` 在测试集上评估性能
2. **推理预测**：使用 `scripts/infer.py` 对新数据进行预测
3. **可视化结果**：查看分割结果的可视化效果

详细信息请参考 [评估指南](EVALUATION_GUIDE.md)。

---

## 技术支持

如有问题，请检查：
- 数据集格式是否正确
- 依赖包是否完整安装
- GPU 驱动和 CUDA 版本是否兼容

祝训练顺利！🚀
