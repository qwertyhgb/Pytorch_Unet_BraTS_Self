"""
BraTS2021 脑肿瘤分割模型推理与可视化脚本

功能：
    1. 加载训练好的 U-Net 2D 模型
    2. 对单张或多张图像进行推理
    3. 可视化输入图像、真实标签和预测结果
    4. 计算评估指标（Dice Score、IoU 等）
    5. 支持保存预测结果和可视化图像
    6. 支持批量推理和结果统计

使用示例：
    # 单张图像推理
    python scripts/infer.py --model checkpoints/best_model.pth --image data_png/imgs/BraTS2021_00000_042.png
    
    # 批量推理
    python scripts/infer.py --model checkpoints/best_model.pth --data_dir data_png --num_samples 10
    
    # 保存结果
    python scripts/infer.py --model checkpoints/best_model.pth --image xxx.png --save_dir results
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path
import glob
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet2d import UNet2D


# ==================== 配置参数 ====================

def parse_args():
    """
    解析命令行参数
    
    该函数定义了所有可用的命令行参数，包括模型配置、数据路径、可视化选项等。
    用户可以通过命令行灵活控制推理行为，无需修改代码。
    
    Returns:
        argparse.Namespace: 包含所有推理配置的参数对象
    """
    parser = argparse.ArgumentParser(description="BraTS2021 U-Net 2D 推理脚本")
    
    # ========== 模型相关参数 ==========
    # 这些参数必须与训练时的模型配置保持一致，否则无法正确加载权重
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                        help="模型权重文件路径（.pth 格式）")
    parser.add_argument("--in_channels", type=int, default=4,
                        help="输入通道数（BraTS 数据集有 4 种 MRI 模态：FLAIR, T1, T1CE, T2）")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="输出类别数（0=背景, 1=坏死/非增强核心, 2=水肿, 3=增强肿瘤）")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="U-Net 基础通道数（影响模型大小和性能）")
    
    # ========== 数据相关参数 ==========
    # 支持两种推理模式：单张图像推理 或 批量推理
    parser.add_argument("--image", type=str, default=None,
                        help="单张图像路径（指定后将进行单张推理，优先级高于 data_dir）")
    parser.add_argument("--mask", type=str, default=None,
                        help="对应的真实标签路径（可选，提供后会计算 Dice/IoU 等评估指标）")
    parser.add_argument("--data_dir", type=str, default="./data_png",
                        help="数据集根目录（用于批量推理，会自动查找 imgs/ 和 masks/ 子目录）")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="批量推理时处理的样本数量（避免一次处理过多图像）")
    parser.add_argument("--random_samples", action="store_true",
                        help="是否随机选择样本（默认 False，按顺序选择前 N 个）")
    
    # ========== 可视化相关参数 ==========
    # 控制如何展示推理结果，支持多种可视化方式
    parser.add_argument("--show_all_modalities", action="store_true",
                        help="是否显示所有 4 种 MRI 模态（FLAIR, T1, T1CE, T2），默认只显示 FLAIR")
    parser.add_argument("--show_overlay", action="store_true",
                        help="是否显示预测结果叠加在原图上（彩色掩码覆盖在灰度图上）")
    parser.add_argument("--colormap", type=str, default="jet",
                        choices=["jet", "viridis", "plasma", "gray"],
                        help="分割结果的颜色映射方案（jet=彩虹色，适合区分类别）")
    parser.add_argument("--figsize", type=int, nargs=2, default=[15, 5],
                        help="matplotlib 图像显示大小，格式：宽 高（单位：英寸）")
    parser.add_argument("--dpi", type=int, default=100,
                        help="图像分辨率（DPI，影响保存图像的清晰度）")
    
    # ========== 保存相关参数 ==========
    # 控制是否保存推理结果和可视化图像
    parser.add_argument("--save_dir", type=str, default=None,
                        help="结果保存目录（不指定则不保存任何文件）")
    parser.add_argument("--save_pred_mask", action="store_true",
                        help="是否保存预测的分割掩码（PNG 格式，像素值 0-3 对应类别）")
    parser.add_argument("--save_visualization", action="store_true",
                        help="是否保存可视化图像（包含输入、真实标签、预测结果的对比图）")
    
    # ========== 其他参数 ==========
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="推理设备（auto=自动检测 GPU，cpu=强制使用 CPU，cuda=强制使用 GPU）")
    parser.add_argument("--no_show", action="store_true",
                        help="不显示图像窗口（适合服务器环境或批量处理，仅保存结果）")
    
    return parser.parse_args()


# ==================== 工具函数 ====================

def load_model(model_path, in_channels=4, num_classes=4, base_channels=64, device="cpu"):
    """
    加载训练好的模型
    
    该函数负责：
    1. 创建与训练时相同架构的模型
    2. 加载保存的权重参数
    3. 处理不同格式的检查点文件（完整检查点 vs 仅权重）
    4. 将模型设置为评估模式并移到指定设备
    
    Args:
        model_path (str): 模型权重文件路径（.pth 文件）
        in_channels (int): 输入通道数（必须与训练时一致）
        num_classes (int): 输出类别数（必须与训练时一致）
        base_channels (int): 基础通道数（必须与训练时一致）
        device (str): 推理设备（'cpu' 或 'cuda'）
    
    Returns:
        nn.Module: 加载好的模型（已设置为评估模式，禁用 dropout 和 batch norm 的训练行为）
    """
    print(f"正在加载模型: {model_path}")
    
    # 检查文件是否存在，避免后续错误
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型实例（空模型，参数随机初始化）
    model = UNet2D(in_ch=in_channels, out_ch=num_classes, base_ch=base_channels)
    
    # 加载权重文件
    # map_location 确保即使在 CPU 上也能加载 GPU 训练的模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的保存格式
    # train.py 中使用 save_checkpoint 保存的是完整检查点（包含优化器、轮数等）
    # 也支持直接保存的模型权重（torch.save(model.state_dict(), path)）
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整的检查点格式（推荐）
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 显示训练时的最佳性能（如果有记录）
        if 'best_dice' in checkpoint:
            print(f"  模型最佳 Dice Score: {checkpoint['best_dice']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  训练轮数: {checkpoint['epoch']}")
    else:
        # 仅包含模型权重的格式
        model.load_state_dict(checkpoint)
    
    # 将模型移到指定设备（CPU 或 GPU）
    model = model.to(device)
    
    # 设置为评估模式：禁用 dropout，batch norm 使用训练时的统计量
    # 这对于推理非常重要，否则结果会不稳定
    model.eval()
    
    # 计算并显示模型参数量（用于了解模型规模）
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    print(f"✓ 模型加载成功")
    
    return model


def load_image(img_path, mask_path=None):
    """
    加载图像和标签
    
    该函数负责从磁盘读取预处理后的 PNG 格式数据：
    - 图像：4 通道 PNG（FLAIR, T1, T1CE, T2），每个通道范围 [0, 255]
    - 标签：单通道 PNG，像素值为 {0, 1, 2, 3} 对应不同类别
    
    Args:
        img_path (str): 图像文件路径（必须是 4 通道 PNG）
        mask_path (str, optional): 标签文件路径（可选，用于评估）
    
    Returns:
        tuple: (img, mask)
            - img: numpy array [H, W, 4]，uint8 类型，范围 [0, 255]
            - mask: numpy array [H, W]，uint8 类型，值为 {0, 1, 2, 3}，或 None
    """
    # 读取 4 通道图像
    # cv2.IMREAD_UNCHANGED 保留原始通道数和数据类型（不会转换为 BGR）
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # 错误处理：文件不存在或无法读取
    if img is None:
        raise IOError(f"无法读取图像: {img_path}")
    
    # 验证图像格式：必须是 [H, W, 4]
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"图像格式错误: 期望 [H, W, 4]，实际 {img.shape}")
    
    # 读取标签（可选）
    mask = None
    if mask_path is not None:
        if os.path.exists(mask_path):
            # 读取为灰度图（单通道）
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠ 警告: 无法读取标签 {mask_path}")
        else:
            print(f"⚠ 警告: 标签文件不存在 {mask_path}")
    
    return img, mask


def preprocess_image(img):
    """
    预处理图像用于模型推理
    
    将 OpenCV 读取的图像转换为 PyTorch 模型所需的格式：
    1. 调整维度顺序（OpenCV 是 HWC，PyTorch 是 CHW）
    2. 归一化到 [0, 1]（与训练时保持一致）
    3. 转换为 Tensor 并添加 batch 维度
    
    Args:
        img (numpy.ndarray): 输入图像 [H, W, 4]，uint8 类型，范围 [0, 255]
    
    Returns:
        torch.Tensor: 预处理后的图像 [1, 4, H, W]，float32 类型，范围 [0, 1]
    """
    # 步骤 1: 转换维度顺序
    # OpenCV 格式: [H, W, C] → PyTorch 格式: [C, H, W]
    img = img.transpose(2, 0, 1).astype(np.float32)
    
    # 步骤 2: 归一化到 [0, 1]
    # 这与 dataset.py 中的预处理保持一致
    img = img / 255.0
    
    # 步骤 3: 转换为 PyTorch Tensor 并添加 batch 维度
    # [C, H, W] → [1, C, H, W]（模型期望 batch 输入）
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, 4, H, W]
    
    return img_tensor


def predict(model, img_tensor, device):
    """
    模型推理
    
    执行前向传播并将模型输出转换为可用的预测结果：
    1. 将输入移到正确的设备（CPU/GPU）
    2. 使用 torch.no_grad() 禁用梯度计算（节省内存，加速推理）
    3. 获取模型输出（logits）并转换为概率和类别预测
    4. 将结果转换为 numpy 数组便于后续处理
    
    Args:
        model (nn.Module): 已加载的 U-Net 模型（评估模式）
        img_tensor (torch.Tensor): 预处理后的输入图像 [1, 4, H, W]
        device (str): 推理设备（'cpu' 或 'cuda'）
    
    Returns:
        tuple: (pred_mask, pred_probs)
            - pred_mask: numpy array [H, W]，每个像素的预测类别（0-3）
            - pred_probs: numpy array [4, H, W]，每个像素属于各类别的概率
    """
    # 将输入张量移到模型所在的设备
    img_tensor = img_tensor.to(device)
    
    # 推理过程
    with torch.no_grad():  # 禁用梯度计算，节省内存并加速
        # 前向传播：获取原始输出（logits，未归一化的分数）
        logits = model(img_tensor)  # [1, 4, H, W]
        
        # 将 logits 转换为概率分布（每个像素的 4 个类别概率和为 1）
        probs = torch.softmax(logits, dim=1)  # [1, 4, H, W]
        
        # 获取每个像素的预测类别（概率最大的类别）
        pred = torch.argmax(logits, dim=1)  # [1, H, W]
    
    # 转换为 numpy 数组并移除 batch 维度
    pred_mask = pred.squeeze(0).cpu().numpy()  # [H, W]
    pred_probs = probs.squeeze(0).cpu().numpy()  # [4, H, W]
    
    return pred_mask, pred_probs


def calculate_metrics(pred_mask, gt_mask, num_classes=4):
    """
    计算评估指标
    
    计算医学图像分割中常用的评估指标：
    - Accuracy: 整体像素准确率
    - Dice Score: 衡量预测和真实区域的重叠程度（F1 Score 的变体）
    - IoU (Intersection over Union): 交并比，也称 Jaccard Index
    
    Dice Score 公式: 2 * |A ∩ B| / (|A| + |B|)
    IoU 公式: |A ∩ B| / |A ∪ B|
    
    Args:
        pred_mask (numpy.ndarray): 预测掩码 [H, W]，值为 {0, 1, 2, 3}
        gt_mask (numpy.ndarray): 真实掩码 [H, W]，值为 {0, 1, 2, 3}
        num_classes (int): 类别数（默认 4）
    
    Returns:
        dict: 包含各种评估指标的字典
            - accuracy: 整体准确率
            - dice_per_class: 每个类别的 Dice Score（列表）
            - iou_per_class: 每个类别的 IoU（列表）
            - mean_dice: 平均 Dice Score（仅肿瘤类别 1, 2, 3）
            - mean_iou: 平均 IoU（仅肿瘤类别 1, 2, 3）
    """
    metrics = {}
    
    # 计算整体准确率（所有像素中预测正确的比例）
    accuracy = (pred_mask == gt_mask).sum() / gt_mask.size
    metrics['accuracy'] = accuracy
    
    # 为每个类别分别计算 Dice Score 和 IoU
    dice_scores = []
    iou_scores = []
    
    for c in range(num_classes):
        # 生成当前类别的二值掩码
        pred_c = (pred_mask == c)  # 预测为类别 c 的像素
        gt_c = (gt_mask == c)      # 真实为类别 c 的像素
        
        # 计算交集和并集
        intersection = (pred_c & gt_c).sum()  # 预测和真实都为 True 的像素数
        union = (pred_c | gt_c).sum()         # 预测或真实为 True 的像素数
        pred_sum = pred_c.sum()               # 预测为类别 c 的总像素数
        gt_sum = gt_c.sum()                   # 真实为类别 c 的总像素数
        
        # 计算 Dice Score
        # 添加平滑项 1e-5 避免除零错误（当某类别不存在时）
        dice = (2.0 * intersection + 1e-5) / (pred_sum + gt_sum + 1e-5)
        dice_scores.append(dice)
        
        # 计算 IoU (Jaccard Index)
        iou = (intersection + 1e-5) / (union + 1e-5)
        iou_scores.append(iou)
    
    # 保存每个类别的指标
    metrics['dice_per_class'] = dice_scores
    metrics['iou_per_class'] = iou_scores
    
    # 计算平均指标（只计算肿瘤类别 1, 2, 3，忽略背景类 0）
    # 这是 BraTS 竞赛的标准评估方式
    metrics['mean_dice'] = np.mean(dice_scores[1:])
    metrics['mean_iou'] = np.mean(iou_scores[1:])
    
    return metrics


def visualize_results(img, gt_mask, pred_mask, pred_probs=None, 
                     show_all_modalities=False, show_overlay=False,
                     colormap='jet', figsize=(15, 5), save_path=None):
    """
    可视化推理结果
    
    创建一个包含多个子图的可视化面板，展示：
    - 输入图像（1 个或 4 个 MRI 模态）
    - 真实标签（如果提供）
    - 预测结果
    - 叠加显示（如果启用）
    
    使用 matplotlib 进行绘图，支持保存为高分辨率图像。
    
    Args:
        img (numpy.ndarray): 输入图像 [H, W, 4]，4 个 MRI 模态
        gt_mask (numpy.ndarray): 真实标签 [H, W]（可为 None，不提供则不显示）
        pred_mask (numpy.ndarray): 预测标签 [H, W]，值为 {0, 1, 2, 3}
        pred_probs (numpy.ndarray): 预测概率 [4, H, W]（当前未使用，预留）
        show_all_modalities (bool): 是否显示所有 4 种 MRI 模态
        show_overlay (bool): 是否显示预测结果叠加在原图上
        colormap (str): matplotlib 颜色映射名称（'jet', 'viridis' 等）
        figsize (tuple): 图像大小（宽, 高），单位英寸
        save_path (str): 保存路径（可选，不提供则不保存）
    
    Returns:
        matplotlib.figure.Figure: 创建的图像对象
    """
    # 定义 MRI 模态名称（BraTS 数据集的 4 种扫描序列）
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
    
    # 定义类别名称和对应的可视化颜色
    class_names = ['背景', '坏死/非增强核心', '水肿', '增强肿瘤']
    class_colors = np.array([
        [0, 0, 0],       # 类别 0 (背景)：黑色
        [255, 0, 0],     # 类别 1 (坏死/非增强核心 NCR/NET)：红色
        [0, 255, 0],     # 类别 2 (水肿 ED)：绿色
        [0, 0, 255],     # 类别 3 (增强肿瘤 ET)：蓝色
    ])
    
    # 计算需要显示的子图数量
    num_plots = 2  # 基础：输入图像 + 预测结果
    if gt_mask is not None:
        num_plots += 1  # 如果有真实标签，添加一个子图
    if show_all_modalities:
        num_plots += 3  # 如果显示所有模态，添加 T1, T1CE, T2（FLAIR 已显示）
    if show_overlay:
        num_plots += 1  # 如果显示叠加图，添加一个子图
    
    # 创建 matplotlib 画布（1 行 N 列的子图布局）
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    # 处理只有一个子图的情况（axes 不是数组）
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0  # 当前子图索引
    
    # 1. 显示输入图像（默认显示 FLAIR 模态）
    # FLAIR 对水肿区域敏感，是脑肿瘤分割中最常用的模态
    axes[plot_idx].imshow(img[:, :, 0], cmap='gray')
    axes[plot_idx].set_title(f'输入图像 ({modality_names[0]})', fontsize=12, fontweight='bold')
    axes[plot_idx].axis('off')  # 隐藏坐标轴
    plot_idx += 1
    
    # 2. 显示其他 MRI 模态（可选）
    # T1: 解剖结构清晰
    # T1CE: 对比增强，突出血脑屏障破坏区域
    # T2: 对水肿敏感
    if show_all_modalities:
        for i in range(1, 4):
            axes[plot_idx].imshow(img[:, :, i], cmap='gray')
            axes[plot_idx].set_title(f'输入图像 ({modality_names[i]})', fontsize=12)
            axes[plot_idx].axis('off')
            plot_idx += 1
    
    # 3. 显示真实标签（如果提供）
    # 用于对比预测结果的准确性
    if gt_mask is not None:
        axes[plot_idx].imshow(gt_mask, cmap=colormap, vmin=0, vmax=3)
        axes[plot_idx].set_title('真实标签 (Ground Truth)', fontsize=12, fontweight='bold')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # 4. 显示预测结果
    # vmin=0, vmax=3 确保颜色映射范围固定（0-3 对应 4 个类别）
    axes[plot_idx].imshow(pred_mask, cmap=colormap, vmin=0, vmax=3)
    axes[plot_idx].set_title('预测结果 (Prediction)', fontsize=12, fontweight='bold')
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # 5. 显示叠加图（可选）
    # 将彩色分割掩码叠加在原始灰度图上，便于直观查看分割区域
    if show_overlay:
        # 创建彩色掩码（RGB 格式）
        colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        
        # 为每个肿瘤类别填充对应颜色（跳过背景类 0）
        for c in range(1, 4):
            colored_mask[pred_mask == c] = class_colors[c]
        
        # 将 FLAIR 图像转换为 RGB 格式（作为底图）
        base_img = cv2.cvtColor((img[:, :, 0]).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # 使用加权混合：60% 原图 + 40% 彩色掩码
        # 这样既能看到解剖结构，又能看到分割结果
        overlay = cv2.addWeighted(base_img, 0.6, colored_mask, 0.4, 0)
        
        axes[plot_idx].imshow(overlay)
        axes[plot_idx].set_title('叠加显示 (Overlay)', fontsize=12, fontweight='bold')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # 调整子图间距，避免标题重叠
    plt.tight_layout()
    
    # 保存可视化结果（如果指定了保存路径）
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化结果已保存: {save_path}")
    
    return fig


# ==================== 主推理流程 ====================

def infer_single_image(model, img_path, mask_path, args, device):
    """
    对单张图像进行推理
    
    完整的单图像推理流程：
    1. 加载图像和标签
    2. 预处理
    3. 模型推理
    4. 计算评估指标（如果有标签）
    5. 可视化结果
    6. 保存结果（如果指定）
    
    Args:
        model (nn.Module): 已加载的模型
        img_path (str): 图像文件路径
        mask_path (str): 标签文件路径（可选，用于评估）
        args (Namespace): 命令行参数对象
        device (str): 推理设备
    """
    print(f"\n{'=' * 60}")
    print(f"推理图像: {os.path.basename(img_path)}")
    print(f"{'=' * 60}")
    
    # 加载图像
    img, gt_mask = load_image(img_path, mask_path)
    print(f"  图像形状: {img.shape}")
    print(f"  图像范围: [{img.min()}, {img.max()}]")
    
    # 预处理
    img_tensor = preprocess_image(img)
    
    # 推理
    pred_mask, pred_probs = predict(model, img_tensor, device)
    print(f"  预测形状: {pred_mask.shape}")
    print(f"  预测类别: {np.unique(pred_mask)}")
    
    # 计算指标（如果有真实标签）
    if gt_mask is not None:
        metrics = calculate_metrics(pred_mask, gt_mask, args.num_classes)
        print(f"\n评估指标:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  平均 Dice Score: {metrics['mean_dice']:.4f}")
        print(f"  平均 IoU: {metrics['mean_iou']:.4f}")
        print(f"  各类别 Dice Score:")
        for i, dice in enumerate(metrics['dice_per_class']):
            print(f"    类别 {i}: {dice:.4f}")
    
    # 可视化
    save_path = None
    if args.save_dir and args.save_visualization:
        os.makedirs(args.save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(args.save_dir, f"{base_name}_visualization.png")
    
    fig = visualize_results(
        img, gt_mask, pred_mask, pred_probs,
        show_all_modalities=args.show_all_modalities,
        show_overlay=args.show_overlay,
        colormap=args.colormap,
        figsize=tuple(args.figsize),
        save_path=save_path
    )
    
    # 保存预测掩码
    if args.save_dir and args.save_pred_mask:
        os.makedirs(args.save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_save_path = os.path.join(args.save_dir, f"{base_name}_pred.png")
        cv2.imwrite(mask_save_path, pred_mask.astype(np.uint8))
        print(f"✓ 预测掩码已保存: {mask_save_path}")
    
    # 显示图像
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def infer_batch(model, data_dir, args, device):
    """
    批量推理
    
    对数据集中的多张图像进行批量推理，并统计整体性能：
    1. 扫描数据目录，查找所有图像文件
    2. 根据参数选择样本（随机或顺序）
    3. 逐张进行推理和评估
    4. 计算并显示平均指标
    5. 可选保存所有结果
    
    Args:
        model (nn.Module): 已加载的模型
        data_dir (str): 数据集根目录
        args (Namespace): 命令行参数对象
        device (str): 推理设备
    """
    print(f"\n{'=' * 60}")
    print(f"批量推理")
    print(f"{'=' * 60}")
    
    # 查找图像文件
    # 支持两种目录结构：
    # 1. data_png/imgs/*.png 和 data_png/masks/*.png（推荐）
    # 2. data_png/*_img.png 和 data_png/*_mask.png（旧版）
    imgs_dir = os.path.join(data_dir, "imgs")
    masks_dir = os.path.join(data_dir, "masks")
    
    if os.path.exists(imgs_dir):
        # 新版目录结构
        img_paths = sorted(glob.glob(os.path.join(imgs_dir, "*.png")))
    else:
        # 旧版命名方式
        img_paths = sorted(glob.glob(os.path.join(data_dir, "*_img.png")))
    
    # 检查是否找到图像
    if len(img_paths) == 0:
        raise ValueError(f"在 {data_dir} 中未找到图像文件")
    
    print(f"找到 {len(img_paths)} 张图像")
    
    # 选择要推理的样本
    if args.random_samples:
        # 随机选择（用于快速测试不同样本）
        import random
        indices = random.sample(range(len(img_paths)), min(args.num_samples, len(img_paths)))
    else:
        # 顺序选择前 N 个（可重复）
        indices = range(min(args.num_samples, len(img_paths)))
    
    selected_paths = [img_paths[i] for i in indices]
    print(f"选择 {len(selected_paths)} 张图像进行推理")
    
    # 批量推理循环
    all_metrics = []  # 存储所有样本的评估指标
    
    # 使用 tqdm 显示进度条
    for img_path in tqdm(selected_paths, desc="推理进度"):
        # 查找对应的标签文件
        if os.path.exists(imgs_dir):
            # 新版结构：imgs/xxx.png → masks/xxx.png
            mask_path = os.path.join(masks_dir, os.path.basename(img_path))
        else:
            # 旧版结构：xxx_img.png → xxx_mask.png
            mask_path = img_path.replace("_img.png", "_mask.png")
        
        # 检查标签文件是否存在
        if not os.path.exists(mask_path):
            mask_path = None
        
        # 对当前图像进行推理
        try:
            # 加载图像和标签
            img, gt_mask = load_image(img_path, mask_path)
            
            # 预处理并推理
            img_tensor = preprocess_image(img)
            pred_mask, pred_probs = predict(model, img_tensor, device)
            
            # 计算评估指标（如果有真实标签）
            if gt_mask is not None:
                metrics = calculate_metrics(pred_mask, gt_mask, args.num_classes)
                all_metrics.append(metrics)
            
            # 可视化和保存结果（如果指定了保存目录）
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # 保存可视化图像
                if args.save_visualization:
                    save_path = os.path.join(args.save_dir, f"{base_name}_visualization.png")
                    fig = visualize_results(
                        img, gt_mask, pred_mask, pred_probs,
                        show_all_modalities=args.show_all_modalities,
                        show_overlay=args.show_overlay,
                        colormap=args.colormap,
                        figsize=tuple(args.figsize),
                        save_path=save_path
                    )
                    plt.close(fig)  # 关闭图像释放内存
                
                # 保存预测掩码（PNG 格式）
                if args.save_pred_mask:
                    mask_save_path = os.path.join(args.save_dir, f"{base_name}_pred.png")
                    cv2.imwrite(mask_save_path, pred_mask.astype(np.uint8))
        
        except Exception as e:
            # 捕获异常，避免单个样本错误导致整个批量推理中断
            print(f"⚠ 处理 {os.path.basename(img_path)} 时出错: {str(e)}")
            continue
    
    # 统计并显示批量推理结果
    if len(all_metrics) > 0:
        print(f"\n{'=' * 60}")
        print(f"批量推理统计结果 (共 {len(all_metrics)} 张图像)")
        print(f"{'=' * 60}")
        
        # 计算所有样本的平均指标
        avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
        avg_dice = np.mean([m['mean_dice'] for m in all_metrics])
        avg_iou = np.mean([m['mean_iou'] for m in all_metrics])
        
        print(f"  平均准确率: {avg_accuracy:.4f}")
        print(f"  平均 Dice Score: {avg_dice:.4f} (仅肿瘤类别)")
        print(f"  平均 IoU: {avg_iou:.4f} (仅肿瘤类别)")
        
        # 显示每个类别的平均 Dice Score
        print(f"\n  各类别平均 Dice Score:")
        class_names = ['背景', '坏死/非增强核心', '水肿', '增强肿瘤']
        for c in range(args.num_classes):
            class_dice = np.mean([m['dice_per_class'][c] for m in all_metrics])
            print(f"    类别 {c} ({class_names[c]}): {class_dice:.4f}")


def main():
    """
    主函数
    
    程序入口，负责：
    1. 解析命令行参数
    2. 设置推理设备（CPU/GPU）
    3. 加载模型
    4. 根据参数选择推理模式（单张 or 批量）
    5. 执行推理
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置推理设备
    # auto: 自动检测（优先使用 GPU）
    # cpu/cuda: 强制使用指定设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 打印欢迎信息
    print(f"{'=' * 60}")
    print(f"BraTS2021 脑肿瘤分割推理")
    print(f"{'=' * 60}")
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    
    # 加载训练好的模型
    model = load_model(
        args.model,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        device=device
    )
    
    # 根据参数选择推理模式
    if args.image is not None:
        # 模式 1: 单张图像推理（指定了 --image 参数）
        infer_single_image(model, args.image, args.mask, args, device)
    else:
        # 模式 2: 批量推理（使用 --data_dir 参数）
        infer_batch(model, args.data_dir, args, device)
    
    # 完成提示
    print(f"\n{'=' * 60}")
    print(f"推理完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
