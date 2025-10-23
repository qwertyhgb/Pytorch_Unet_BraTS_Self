# 🧠 BraTS2021 U-Net(2D) 多模态脑肿瘤分割

本项目使用 **U-Net(2D)** 模型，对 **BraTS2021 多模态 MRI 数据集** 进行脑肿瘤分割。  
实现从 `.nii.gz` 医学图像到 `.png` 切片，再到 PyTorch 训练、评估与可视化的完整流程。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 目录

- [项目特点](#-项目特点)
- [环境配置](#-环境配置)
- [数据准备](#-数据准备)
- [项目结构](#-项目结构)
- [使用方法](#-使用方法)
  - [数据预处理](#1-数据预处理)
  - [模型训练](#2-模型训练)
  - [模型推理](#3-模型推理)
- [模型架构](#-模型架构)
- [评估指标](#-评估指标)
- [实验结果](#-实验结果)
- [常见问题](#-常见问题)
- [参考资料](#-参考资料)

---

## ✨ 项目特点

- 🎯 **完整流程**：从数据预处理到模型训练、推理的端到端实现
- 🏗️ **经典架构**：基于 U-Net 的医学图像分割网络，带详细中文注释
- 📊 **多种评估**：支持 Dice Score、IoU、Accuracy 等多种评估指标
- 🎨 **可视化**：丰富的可视化功能，支持多模态显示和结果叠加
- ⚙️ **灵活配置**：命令行参数支持，无需修改代码即可调整配置
- 📦 **模块化设计**：代码结构清晰，易于扩展和修改
- 🔧 **实用工具**：包含数据增强、早停、学习率调度等训练技巧

---

## 🚀 环境配置

### 1. 创建虚拟环境

```bash
# 使用 conda
conda create -n brats2d python=3.9
conda activate brats2d

# 或使用 venv
python -m venv brats2d
source brats2d/bin/activate  # Linux/Mac
# 或
brats2d\Scripts\activate  # Windows
```

## ⚡ 快速开始（Windows - PowerShell）

以下是一个从环境准备到训练、推理的最小可复制流程，适用于 Windows PowerShell（已假定仓库根目录为当前工作目录）。

1. 创建并激活虚拟环境（PowerShell）:

```powershell
# 创建虚拟环境
python -m venv .venv

# 激活（PowerShell）
.\.venv\Scripts\Activate.ps1
```

2. 安装依赖:

```powershell
pip install -r requirements.txt
```

3. 数据准备（示例）:

```powershell
# 假设你已将 BraTS 数据集放在 data\BraTS2021 文件夹下
# 将原始 .nii.gz 文件转换为 png 切片，输出到 data_png（脚本：scripts/convert_brats_to_png.py）
python .\scripts\convert_brats_to_png.py --input .\data\BraTS2021 --output .\data_png
```

注意：不同版本的转换脚本参数可能不同。上面的命令为示例，若脚本使用不同参数，请运行 `python .\scripts\convert_brats_to_png.py -h` 查看帮助。

## 🗂 数据准备细节

- 输入：原始 BraTS 数据（每个病人一个文件夹，包含多个模态的 .nii.gz）
- 输出：按切片保存的 PNG 图像与对应的 mask，默认路径为 `data_png/imgs` 与 `data_png/masks`。

检查点：转换完成后，建议检查 `data_png` 中的样例图像与掩码是否对齐（同一文件名前缀、尺寸一致）。

## 🧠 训练示例（示范命令）

下面给出一个训练示例命令 —— 这是一个通用示例，假设 `train.py` 支持常见命令行参数（`--data`, `--epochs`, `--batch-size`, `--lr`, `--device`, `--save-dir`）。如果脚本使用其他参数，请用 `-h` 查看具体选项并替换。

```powershell
python .\train.py --data .\data_png --epochs 50 --batch-size 8 --lr 1e-3 --device cuda:0 --save-dir .\checkpoints
```

训练过程中会在 `--save-dir` 指定目录保存模型检查点（例如 `best.pth`、`last.pth` 等）。

## 🚀 推理 / 可视化（示例）

假设仓库包含 `scripts/infer.py`，下面示例演示如何使用训练好的模型对切片或体积进行推理：

```powershell
python .\scripts\infer.py --model .\checkpoints\best.pth --input .\data_png\imgs --output .\outputs --device cuda:0
```

输出将保存在 `outputs` 文件夹下，包含预测 mask 与可视化结果（如果脚本支持）。同样，若脚本参数不同，请先运行 `python .\scripts\infer.py -h` 查看。

## 📦 依赖说明

- 已列在 `requirements.txt` 中。典型依赖包括：
  - Python 3.8+
  - torch, torchvision
  - numpy, pillow, scikit-image, nibabel（用于 .nii.gz 读写，如果使用）

在 GPU 环境下安装带 CUDA 的 PyTorch：

```powershell
# 示例：安装适用于 CUDA 11.8 的 PyTorch（请根据官方说明选择合适索引）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## ❓ 常见问题 / 故障排查

- 转换脚本运行报错（找不到 nibabel 或 SimpleITK）：
  - 解决：`pip install nibabel SimpleITK` 或检查 `requirements.txt` 并安装所有依赖。

- 训练时 CUDA 不可用：
  - 检查 GPU 驱动与 CUDA 版本是否匹配。
  - 在 Python 中运行：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

- 结果不好 / Dice 分数低：
  - 检查数据预处理（归一化、切片顺序、标签映射）是否与训练脚本一致。
  - 尝试数据增强、学习率调整或更长训练。

## 🔧 开发者提示

- 若你要修改模型架构，可在 `models/unet2d.py` 中进行；训练逻辑主要在 `train.py`，推理相关脚本在 `scripts/` 下。
- 添加新依赖后，更新 `requirements.txt` 并在 README 中记录。

## ✅ 完成小结

本次更新补充了一个面向 Windows PowerShell 用户的“快速开始”节，示例命令包含虚拟环境、依赖安装、数据转换、训练与推理的示范命令。命令为示例，某些脚本参数可能需根据各自脚本的 `-h` 输出进行调整。

### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果需要 GPU 支持（推荐）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
