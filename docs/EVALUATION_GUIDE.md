# 🎯 BraTS2021 模型评估完整指南

本指南详细介绍如何评估训练好的 U-Net 2D 脑肿瘤分割模型。

---

## 📋 目录

- [准备工作](#准备工作)
- [快速开始](#快速开始)
- [评估模式](#评估模式)
- [命令行参数](#命令行参数)
- [输出结果](#输出结果)
- [评估指标解读](#评估指标解读)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)

---

## 🔧 准备工作

### 1. 检查必需文件

确保以下文件和目录存在：

```bash
项目根目录/
├── checkpoints/
│   └── best_model.pth          # 训练好的模型权重
├── data_png/                    # 测试数据集
│   ├── imgs/                    # 图像文件夹
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── ...
│   └── masks/                   # 标签文件夹
│       ├── sample_001.png
│       ├── sample_002.png
│       └── ...
├── scripts/
│   └── evaluate.py              # 评估脚本
└── models/
    └── unet2d.py                # 模型定义
```

### 2. 验证环境

```bash
# 检查 Python 版本
python --version  # 应该是 3.8+

# 检查 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 检查 CUDA 可用性（如果使用 GPU）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 检查数据集

```bash
# Windows
dir data_png\imgs
dir data_png\masks

# Linux/Mac
ls data_png/imgs/
ls data_png/masks/
```

---

## 🚀 快速开始

### 最简单的评估命令

```bash
python scripts/evaluate.py --model checkpoints/best_model.pth --data_dir data_png
```

**这会做什么？**
- ✅ 加载训练好的模型
- ✅ 在测试集上进行推理
- ✅ 计算评估指标
- ✅ 在终端显示结果

**预期输出：**
```
============================================================
BraTS2021 脑肿瘤分割模型评估
============================================================
使用设备: cuda
GPU 型号: NVIDIA GeForce RTX 3090

正在加载模型: checkpoints/best_model.pth
  训练时最佳 Dice Score: 0.8234
  训练轮数: 30
✓ 模型加载成功

正在加载数据集: data_png
✓ 数据集加载成功，共 1000 个样本

开始评估...
评估进度: 100%|████████████████| 125/125 [00:45<00:00,  2.78it/s]

正在计算评估指标...

============================================================
评估结果
============================================================

整体性能（平均，忽略背景类）:
  Dice Score:  0.8234
  IoU:         0.7123
  Precision:   0.8456
  Recall:      0.8012
  F1 Score:    0.8228
  整体准确率:  0.9567

各类别 Dice Score:
  背景        : 0.9876
  坏死/非增强核心: 0.7845
  水肿        : 0.8523
  增强肿瘤    : 0.8334

============================================================
评估完成！
============================================================
```

---

## 📊 评估模式

### 模式 1：基础评估（仅显示指标）

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png
```

**适用场景：** 快速查看模型性能

---

### 模式 2：保存详细报告

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --save_dir results/evaluation
```

**生成文件：**
```
results/evaluation/
├── evaluation_metrics.json      # JSON 格式完整指标
├── evaluation_metrics.csv       # CSV 表格（可用 Excel 打开）
├── evaluation_report.md         # Markdown 格式报告
└── confusion_matrix.png         # 混淆矩阵可视化
```

**适用场景：** 需要保存评估结果用于论文或报告

---

### 模式 3：可视化预测结果

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --visualize \
    --num_samples 10
```

**效果：**
- 🖼️ 随机选择 10 个样本
- 🎨 显示输入图像（多个 MRI 模态）
- 📊 显示真实标签 vs 预测结果
- 🔍 显示叠加效果
- 📈 显示每个样本的 Dice Score

**适用场景：** 直观查看模型预测效果

---

### 模式 4：完整评估（推荐）⭐

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --save_dir results/evaluation \
    --visualize \
    --num_samples 10 \
    --batch_size 8
```

**包含所有功能：**
- ✅ 计算所有评估指标
- ✅ 生成详细报告（JSON、CSV、Markdown）
- ✅ 绘制混淆矩阵
- ✅ 可视化预测结果
- ✅ 保存所有输出

**适用场景：** 完整的模型评估和结果展示

---

## ⚙️ 命令行参数

### 模型相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | **必需** | 模型权重文件路径 |
| `--in_channels` | int | 4 | 输入通道数（MRI 模态数） |
| `--num_classes` | int | 4 | 输出类别数 |
| `--base_channels` | int | 64 | U-Net 基础通道数 |

### 数据相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | **必需** | 测试数据集目录 |
| `--batch_size` | int | 8 | 批次大小 |
| `--num_workers` | int | 4 | 数据加载线程数 |

### 评估相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save_dir` | str | None | 结果保存目录 |
| `--visualize` | flag | False | 是否可视化结果 |
| `--num_samples` | int | 5 | 可视化样本数量 |
| `--save_predictions` | flag | False | 是否保存所有预测 |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--device` | str | auto | 推理设备（auto/cpu/cuda） |

### 参数使用示例

```bash
# 使用 CPU 评估
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --device cpu

# 调整批次大小（显存不足时）
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --batch_size 4

# 可视化更多样本
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --visualize \
    --num_samples 20

# 在验证集上评估
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png_val \
    --save_dir results/val_evaluation
```

---

## 📁 输出结果

### 1. JSON 格式指标 (`evaluation_metrics.json`)

```json
{
    "class_metrics": {
        "class_0": {
            "dice": 0.9876,
            "iou": 0.9754,
            "precision": 0.9890,
            "recall": 0.9862,
            ...
        },
        ...
    },
    "average_metrics": {
        "mean_dice": 0.8234,
        "mean_iou": 0.7123,
        ...
    }
}
```

**用途：** 程序读取、进一步分析

---

### 2. CSV 格式指标 (`evaluation_metrics.csv`)

| 类别 | Dice | IoU | Precision | Recall | F1 | Specificity | Accuracy |
|------|------|-----|-----------|--------|----|--------------|-----------| 
| 背景 | 0.9876 | 0.9754 | 0.9890 | 0.9862 | 0.9876 | 0.9998 | 0.9987 |
| 坏死/非增强核心 | 0.7845 | 0.6456 | 0.8123 | 0.7589 | 0.7845 | 0.9876 | 0.9654 |
| 水肿 | 0.8523 | 0.7423 | 0.8756 | 0.8301 | 0.8523 | 0.9912 | 0.9789 |
| 增强肿瘤 | 0.8334 | 0.7156 | 0.8589 | 0.8089 | 0.8334 | 0.9934 | 0.9823 |

**用途：** Excel 打开、制作表格

---

### 3. Markdown 报告 (`evaluation_report.md`)

```markdown
# BraTS2021 脑肿瘤分割模型评估报告

**评估时间**: 2024-01-15 14:30:25

## 整体性能

- **平均 Dice Score**: 0.8234
- **平均 IoU**: 0.7123
- **平均 Precision**: 0.8456
...

## 各类别详细指标

| 类别 | Dice | IoU | ... |
|------|------|-----|-----|
...
```

**用途：** 文档展示、报告撰写

---

### 4. 混淆矩阵 (`confusion_matrix.png`)

显示两种形式：
- **左图：** 绝对数量（像素数）
- **右图：** 归一化百分比

**用途：** 分析模型在各类别间的混淆情况

---

### 5. 可视化结果 (`visualizations/`)

每个样本生成一张图，包含：
- **第一行：** FLAIR、T1、T1CE 三种 MRI 模态
- **第二行：** 真实标签、预测结果、叠加显示

**用途：** 直观展示模型预测效果

---

## 📈 评估指标解读

### Dice Score（最重要）

| 分数范围 | 评价 | 说明 |
|----------|------|------|
| > 0.9 | 优秀 ⭐⭐⭐ | 模型性能非常好 |
| 0.8 - 0.9 | 良好 ⭐⭐ | 模型性能较好 |
| 0.7 - 0.8 | 一般 ⭐ | 模型性能中等 |
| 0.6 - 0.7 | 较差 ⚠️ | 需要改进 |
| < 0.6 | 很差 ❌ | 需要重新训练 |

### 各指标含义

| 指标 | 公式 | 含义 | 范围 |
|------|------|------|------|
| **Dice Score** | 2×TP/(2×TP+FP+FN) | 预测和真实区域的重叠程度 | [0, 1] |
| **IoU** | TP/(TP+FP+FN) | 交并比 | [0, 1] |
| **Precision** | TP/(TP+FP) | 预测为正例中真正为正例的比例 | [0, 1] |
| **Recall** | TP/(TP+FN) | 真实正例中被正确预测的比例 | [0, 1] |
| **F1 Score** | 2×P×R/(P+R) | Precision 和 Recall 的调和平均 | [0, 1] |
| **Specificity** | TN/(TN+FP) | 真实负例中被正确预测的比例 | [0, 1] |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 所有预测正确的比例 | [0, 1] |

### 指标选择建议

- **医学图像分割：** 优先关注 **Dice Score** 和 **Recall**
- **类别不平衡：** 关注 **Precision** 和 **Recall** 的平衡
- **整体性能：** 参考 **F1 Score**

---

## ❓ 常见问题

### Q1: 找不到模型文件

**错误信息：**
```
FileNotFoundError: 模型文件不存在: checkpoints/best_model.pth
```

**解决方法：**
```bash
# 检查模型是否存在
ls checkpoints/

# 如果不存在，需要先训练模型
python train.py --data_dir data_png --epochs 30 --save_dir checkpoints
```

---

### Q2: 找不到数据集

**错误信息：**
```
ValueError: 在 data_png/imgs 中未找到任何图像文件！
```

**解决方法：**
```bash
# 检查数据目录
ls data_png/imgs/
ls data_png/masks/

# 如果不存在，需要先预处理数据
python scripts/convert_brats_to_png.py \
    --input_dir data \
    --output_dir data_png
```

---

### Q3: CUDA out of memory（显存不足）

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方法：**

**方法 1：** 减小批次大小
```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --batch_size 4  # 或更小
```

**方法 2：** 使用 CPU
```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --device cpu
```

---

### Q4: 模型加载失败

**错误信息：**
```
RuntimeError: Error(s) in loading state_dict
```

**可能原因：**
- 模型架构参数不匹配
- 模型文件损坏

**解决方法：**
```bash
# 检查模型参数是否匹配
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --in_channels 4 \
    --num_classes 4 \
    --base_channels 64
```

---

### Q5: 评估速度太慢

**优化方法：**

1. **增加批次大小**（如果显存允许）
```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --batch_size 16
```

2. **增加数据加载线程**
```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --num_workers 8
```

3. **使用 GPU**
```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --device cuda
```

---

## 💡 最佳实践

### 1. 评估前检查清单

- [ ] 模型文件存在且完整
- [ ] 测试数据集准备好
- [ ] 环境配置正确（PyTorch、CUDA）
- [ ] 有足够的磁盘空间保存结果

### 2. 推荐的评估流程

```bash
# 步骤 1: 快速评估（查看整体性能）
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png

# 步骤 2: 保存详细报告
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --save_dir results/evaluation

# 步骤 3: 可视化分析
python scripts/evaluate.py \
    --model checkpoints/best_model.pth \
    --data_dir data_png \
    --save_dir results/evaluation \
    --visualize \
    --num_samples 20
```

### 3. 结果分析建议

1. **查看整体 Dice Score**
   - 是否达到预期目标？
   - 与训练时的验证集 Dice 是否一致？

2. **分析各类别性能**
   - 哪个类别表现最好/最差？
   - 是否存在类别不平衡问题？

3. **检查混淆矩阵**
   - 哪些类别容易混淆？
   - 是否有系统性错误？

4. **查看可视化结果**
   - 模型在哪些情况下表现好/差？
   - 是否有明显的失败案例？

### 4. 性能优化建议

如果评估结果不理想：

- **Dice < 0.7：** 考虑重新训练，调整超参数
- **某类别 Dice 很低：** 使用类别权重或数据增强
- **过拟合：** 增加正则化、数据增强
- **欠拟合：** 增加模型容量、训练更多轮

---

## 📞 获取帮助

如果遇到问题：

1. 查看本指南的[常见问题](#常见问题)部分
2. 检查终端的错误信息
3. 查看项目 README.md
4. 提交 Issue 到项目仓库

---

## 📚 相关文档

- [训练指南](TRAINING_GUIDE.md)
- [数据预处理指南](DATA_PREPROCESSING.md)
- [模型推理指南](INFERENCE_GUIDE.md)
- [项目 README](../README.md)

---

**最后更新：** 2024-01-15  
**版本：** 1.0.0
