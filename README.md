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

