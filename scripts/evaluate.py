"""
BraTS2021 脑肿瘤分割模型评估脚本

功能：
    1. 在测试集上评估训练好的模型
    2. 计算详细的评估指标（Dice Score、IoU、Accuracy、Precision、Recall、F1）
    3. 生成混淆矩阵
    4. 计算各类别的性能指标
    5. 生成完整的评估报告（JSON、CSV、Markdown）
    6. 可视化评估结果

使用示例：
    # 基础评估
    python scripts/evaluate.py --model checkpoints/best_model.pth --data_dir data_png
    
    # 保存详细报告
    python scripts/evaluate.py --model checkpoints/best_model.pth --data_dir data_png --save_dir results/evaluation
    
    # 可视化结果
    python scripts/evaluate.py --model checkpoints/best_model.pth --data_dir data_png --visualize --num_samples 10
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet2d import UNet2D
from dataset import BraTSDataset
from torch.utils.data import DataLoader


# ==================== 配置参数 ====================

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有评估配置的参数对象
    """
    parser = argparse.ArgumentParser(description="BraTS2021 U-Net 2D 模型评估脚本")
    
    # ========== 模型相关 ==========
    parser.add_argument("--model", type=str, required=True,
                        help="模型权重文件路径（.pth 格式）")
    parser.add_argument("--in_channels", type=int, default=4,
                        help="输入通道数（4 种 MRI 模态）")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="输出类别数（背景 + 3 类肿瘤）")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="U-Net 基础通道数")
    
    # ========== 数据相关 ==========
    parser.add_argument("--data_dir", type=str, required=True,
                        help="测试数据集目录")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    
    # ========== 评估相关 ==========
    parser.add_argument("--save_dir", type=str, default=None,
                        help="评估结果保存目录")
    parser.add_argument("--visualize", action="store_true",
                        help="是否可视化评估结果")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="可视化的样本数量")
    parser.add_argument("--save_predictions", action="store_true",
                        help="是否保存所有预测结果")
    
    # ========== 其他 ==========
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="评估设备")
    
    return parser.parse_args()


# ==================== 评估指标计算 ====================

def calculate_metrics_per_class(pred, target, num_classes=4):
    """
    计算每个类别的详细评估指标
    
    包括：Dice Score、IoU、Precision、Recall、F1、Specificity、Accuracy
    以及混淆矩阵的四个基本元素：TP、FP、FN、TN
    
    Args:
        pred (numpy.ndarray): 预测结果 [H, W]，值为 {0, 1, 2, 3}
        target (numpy.ndarray): 真实标签 [H, W]，值为 {0, 1, 2, 3}
        num_classes (int): 类别数（默认 4）
    
    Returns:
        dict: 每个类别的指标字典
    """
    metrics = {}
    
    for c in range(num_classes):
        # 生成二值掩码（当前类别 vs 其他类别）
        pred_c = (pred == c)
        target_c = (target == c)
        
        # 计算混淆矩阵的四个基本元素
        # TP (True Positive): 预测为正，实际为正
        tp = np.sum(pred_c & target_c)
        
        # FP (False Positive): 预测为正，实际为负
        fp = np.sum(pred_c & ~target_c)
        
        # FN (False Negative): 预测为负，实际为正
        fn = np.sum(~pred_c & target_c)
        
        # TN (True Negative): 预测为负，实际为负
        tn = np.sum(~pred_c & ~target_c)
        
        # 添加平滑项避免除零
        eps = 1e-7
        
        # Dice Score = 2 * TP / (2 * TP + FP + FN)
        # 衡量预测和真实区域的重叠程度
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        
        # IoU (Intersection over Union) = TP / (TP + FP + FN)
        # 也称 Jaccard Index，交并比
        iou = (tp + eps) / (tp + fp + fn + eps)
        
        # Precision (精确率) = TP / (TP + FP)
        # 预测为正例中真正为正例的比例
        precision = (tp + eps) / (tp + fp + eps)
        
        # Recall (召回率/灵敏度) = TP / (TP + FN)
        # 真实正例中被正确预测的比例
        recall = (tp + eps) / (tp + fn + eps)
        
        # F1 Score = 2 * Precision * Recall / (Precision + Recall)
        # Precision 和 Recall 的调和平均
        f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
        
        # Specificity (特异性) = TN / (TN + FP)
        # 真实负例中被正确预测的比例
        specificity = (tn + eps) / (tn + fp + eps)
        
        # Accuracy (准确率) = (TP + TN) / (TP + TN + FP + FN)
        # 所有预测正确的比例
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        
        # 保存指标
        metrics[f'class_{c}'] = {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        }
    
    return metrics


def calculate_overall_metrics(all_preds, all_targets, num_classes=4):
    """
    计算整体评估指标
    
    合并所有批次的预测结果，计算全局指标
    
    Args:
        all_preds (list): 所有预测结果列表，每个元素为 [B, H, W]
        all_targets (list): 所有真实标签列表，每个元素为 [B, H, W]
        num_classes (int): 类别数
    
    Returns:
        dict: 包含类别指标和平均指标的字典
    """
    # 合并所有批次的数据
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算每个类别的指标
    class_metrics = calculate_metrics_per_class(all_preds, all_targets, num_classes)
    
    # 计算平均指标（忽略背景类 0）
    # 在医学图像分割中，通常只关注前景类别的性能
    metrics_to_avg = ['dice', 'iou', 'precision', 'recall', 'f1', 'specificity']
    avg_metrics = {}
    
    for metric in metrics_to_avg:
        # 只计算类别 1, 2, 3 的平均值
        values = [class_metrics[f'class_{c}'][metric] for c in range(1, num_classes)]
        avg_metrics[f'mean_{metric}'] = float(np.mean(values))
    
    # 整体准确率（包含所有类别）
    overall_accuracy = float(np.mean(all_preds == all_targets))
    avg_metrics['overall_accuracy'] = overall_accuracy
    
    return {
        'class_metrics': class_metrics,
        'average_metrics': avg_metrics,
    }


# ==================== 混淆矩阵 ====================

def plot_confusion_matrix(all_preds, all_targets, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    显示两种形式：
    1. 绝对数量：每个单元格显示像素数量
    2. 归一化：每个单元格显示百分比（按行归一化）
    
    Args:
        all_preds (list): 所有预测结果列表
        all_targets (list): 所有真实标签列表
        class_names (list): 类别名称列表
        save_path (str, optional): 保存路径
    
    Returns:
        numpy.ndarray: 混淆矩阵
    """
    # 合并并展平数据
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 归一化混淆矩阵（按行归一化，显示每个真实类别的预测分布）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 绝对数量混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': '像素数量'})
    axes[0].set_xlabel('预测类别', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('真实类别', fontsize=12, fontweight='bold')
    axes[0].set_title('混淆矩阵（绝对数量）', fontsize=14, fontweight='bold')
    
    # 2. 归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': '百分比'})
    axes[1].set_xlabel('预测类别', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('真实类别', fontsize=12, fontweight='bold')
    axes[1].set_title('混淆矩阵（归一化）', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {save_path}")
    
    plt.show()
    
    return cm


# ==================== 可视化 ====================

def visualize_predictions(model, dataset, indices, device, save_dir=None):
    """
    可视化预测结果
    
    为每个样本创建一个包含以下内容的可视化面板：
    - 第一行：3 个 MRI 模态（FLAIR, T1, T1CE）
    - 第二行：真实标签、预测结果、叠加显示
    
    Args:
        model (nn.Module): 已加载的模型
        dataset (Dataset): 数据集
        indices (list): 要可视化的样本索引列表
        device (str): 推理设备
        save_dir (str, optional): 保存目录
    """
    model.eval()
    
    # 类别名称和颜色定义
    class_names = ['背景', '坏死/非增强核心', '水肿', '增强肿瘤']
    class_colors = np.array([
        [0, 0, 0],       # 背景：黑色
        [255, 0, 0],     # 类别1：红色
        [0, 255, 0],     # 类别2：绿色
        [0, 0, 255],     # 类别3：蓝色
    ])
    
    for idx in indices:
        # 获取数据
        img_tensor, mask_tensor = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # 转换为 numpy 数组
        mask = mask_tensor.numpy()
        img = img_tensor.squeeze(0).cpu().numpy()
        
        # 计算该样本的指标
        metrics = calculate_metrics_per_class(pred, mask, num_classes=4)
        mean_dice = np.mean([metrics[f'class_{c}']['dice'] for c in range(1, 4)])
        
        # 创建可视化面板
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行：输入图像（3个模态）
        modality_names = ['FLAIR', 'T1', 'T1CE']
        for i in range(3):
            axes[0, i].imshow(img[i], cmap='gray')
            axes[0, i].set_title(f'输入 - {modality_names[i]}', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
        
        # 第二行：真实标签、预测结果、叠加对比
        # 1. 真实标签
        axes[1, 0].imshow(mask, cmap='jet', vmin=0, vmax=3)
        axes[1, 0].set_title('真实标签', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 2. 预测结果
        axes[1, 1].imshow(pred, cmap='jet', vmin=0, vmax=3)
        axes[1, 1].set_title(f'预测结果\nDice: {mean_dice:.4f}', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # 3. 叠加显示
        # 创建彩色掩码
        colored_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for c in range(1, 4):
            colored_mask[pred == c] = class_colors[c]
        
        # 将 FLAIR 图像转换为 RGB
        base_img = cv2.cvtColor((img[0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # 叠加：60% 原图 + 40% 彩色掩码
        overlay = cv2.addWeighted(base_img, 0.6, colored_mask, 0.4, 0)
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('叠加显示', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        # 设置总标题
        plt.suptitle(f'样本 {idx} 评估结果', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'sample_{idx}_visualization.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 可视化已保存: {save_path}")
        
        plt.show()


# ==================== 报告生成 ====================

def generate_report(metrics, save_dir):
    """
    生成评估报告
    
    生成三种格式的报告：
    1. JSON 格式：完整的指标数据，便于程序读取
    2. CSV 格式：表格形式，便于 Excel 打开
    3. Markdown 格式：可读性强，便于文档展示
    
    Args:
        metrics (dict): 评估指标字典
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    class_names = ['背景', '坏死/非增强核心', '水肿', '增强肿瘤']
    
    # ========== 1. 保存 JSON 格式 ==========
    json_path = os.path.join(save_dir, 'evaluation_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"✓ JSON 报告已保存: {json_path}")
    
    # ========== 2. 保存 CSV 格式 ==========
    csv_path = os.path.join(save_dir, 'evaluation_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['类别', 'Dice', 'IoU', 'Precision', 'Recall', 'F1', 'Specificity', 'Accuracy'])
        
        # 写入每个类别的指标
        for c in range(4):
            class_metric = metrics['class_metrics'][f'class_{c}']
            writer.writerow([
                class_names[c],
                f"{class_metric['dice']:.4f}",
                f"{class_metric['iou']:.4f}",
                f"{class_metric['precision']:.4f}",
                f"{class_metric['recall']:.4f}",
                f"{class_metric['f1']:.4f}",
                f"{class_metric['specificity']:.4f}",
                f"{class_metric['accuracy']:.4f}",
            ])
        
        # 写入平均指标
        writer.writerow([])
        writer.writerow(['平均指标（忽略背景）'])
        avg_metrics = metrics['average_metrics']
        writer.writerow([
            '平均',
            f"{avg_metrics['mean_dice']:.4f}",
            f"{avg_metrics['mean_iou']:.4f}",
            f"{avg_metrics['mean_precision']:.4f}",
            f"{avg_metrics['mean_recall']:.4f}",
            f"{avg_metrics['mean_f1']:.4f}",
            f"{avg_metrics['mean_specificity']:.4f}",
            f"{avg_metrics['overall_accuracy']:.4f}",
        ])
    
    print(f"✓ CSV 报告已保存: {csv_path}")
    
    # ========== 3. 生成 Markdown 报告 ==========
    md_path = os.path.join(save_dir, 'evaluation_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# BraTS2021 脑肿瘤分割模型评估报告\n\n")
        f.write(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 整体性能\n\n")
        avg_metrics = metrics['average_metrics']
        f.write(f"- **平均 Dice Score**: {avg_metrics['mean_dice']:.4f}\n")
        f.write(f"- **平均 IoU**: {avg_metrics['mean_iou']:.4f}\n")
        f.write(f"- **平均 Precision**: {avg_metrics['mean_precision']:.4f}\n")
        f.write(f"- **平均 Recall**: {avg_metrics['mean_recall']:.4f}\n")
        f.write(f"- **平均 F1 Score**: {avg_metrics['mean_f1']:.4f}\n")
        f.write(f"- **整体准确率**: {avg_metrics['overall_accuracy']:.4f}\n\n")
        
        f.write("## 各类别详细指标\n\n")
        f.write("| 类别 | Dice | IoU | Precision | Recall | F1 | Specificity | Accuracy |\n")
        f.write("|------|------|-----|-----------|--------|----|--------------|-----------|\n")
        
        for c in range(4):
            class_metric = metrics['class_metrics'][f'class_{c}']
            f.write(f"| {class_names[c]} | "
                   f"{class_metric['dice']:.4f} | "
                   f"{class_metric['iou']:.4f} | "
                   f"{class_metric['precision']:.4f} | "
                   f"{class_metric['recall']:.4f} | "
                   f"{class_metric['f1']:.4f} | "
                   f"{class_metric['specificity']:.4f} | "
                   f"{class_metric['accuracy']:.4f} |\n")
        
        f.write("\n## 指标说明\n\n")
        f.write("- **Dice Score**: 衡量预测和真实区域的重叠程度，范围 [0, 1]，越大越好\n")
        f.write("- **IoU (Intersection over Union)**: 交并比，范围 [0, 1]，越大越好\n")
        f.write("- **Precision**: 精确率，预测为正例中真正为正例的比例\n")
        f.write("- **Recall**: 召回率，真实正例中被正确预测的比例\n")
        f.write("- **F1 Score**: Precision 和 Recall 的调和平均\n")
        f.write("- **Specificity**: 特异性，真实负例中被正确预测的比例\n")
        f.write("- **Accuracy**: 准确率，所有预测正确的比例\n")
    
    print(f"✓ Markdown 报告已保存: {md_path}")


# ==================== 主评估流程 ====================

def evaluate(model, dataloader, device, num_classes=4):
    """
    评估模型
    
    遍历整个数据集，收集所有预测结果和真实标签
    
    Args:
        model (nn.Module): 已加载的模型
        dataloader (DataLoader): 数据加载器
        device (str): 推理设备
        num_classes (int): 类别数
    
    Returns:
        tuple: (all_preds, all_targets)
            - all_preds: 所有预测结果列表
            - all_targets: 所有真实标签列表
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("\n开始评估...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="评估进度"):
            # 将数据移到设备
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 收集结果（转换为 numpy 数组）
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    return all_preds, all_targets


def main():
    """
    主函数
    
    完整的评估流程：
    1. 加载模型和数据
    2. 执行评估
    3. 计算指标
    4. 生成报告
    5. 可视化结果
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置推理设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("BraTS2021 脑肿瘤分割模型评估")
    print("=" * 60)
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    
    # ========== 加载模型 ==========
    print(f"\n正在加载模型: {args.model}")
    model = UNet2D(in_ch=args.in_channels, out_ch=args.num_classes, base_ch=args.base_channels)
    
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_dice' in checkpoint:
            print(f"  训练时最佳 Dice Score: {checkpoint['best_dice']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  训练轮数: {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✓ 模型加载成功")
    
    # ========== 加载数据集 ==========
    print(f"\n正在加载数据集: {args.data_dir}")
    dataset = BraTSDataset(args.data_dir, transform=None, validate_data=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    print(f"✓ 数据集加载成功，共 {len(dataset)} 个样本")
    
    # ========== 执行评估 ==========
    all_preds, all_targets = evaluate(model, dataloader, device, args.num_classes)
    
    # ========== 计算指标 ==========
    print("\n正在计算评估指标...")
    metrics = calculate_overall_metrics(all_preds, all_targets, args.num_classes)
    
    # ========== 打印结果 ==========
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    avg_metrics = metrics['average_metrics']
    print(f"\n整体性能（平均，忽略背景类）:")
    print(f"  Dice Score:  {avg_metrics['mean_dice']:.4f}")
    print(f"  IoU:         {avg_metrics['mean_iou']:.4f}")
    print(f"  Precision:   {avg_metrics['mean_precision']:.4f}")
    print(f"  Recall:      {avg_metrics['mean_recall']:.4f}")
    print(f"  F1 Score:    {avg_metrics['mean_f1']:.4f}")
    print(f"  整体准确率:  {avg_metrics['overall_accuracy']:.4f}")
    
    print(f"\n各类别 Dice Score:")
    class_names = ['背景', '坏死/非增强核心', '水肿', '增强肿瘤']
    for c in range(args.num_classes):
        dice = metrics['class_metrics'][f'class_{c}']['dice']
        print(f"  {class_names[c]:12s}: {dice:.4f}")
    
    # ========== 保存结果 ==========
    if args.save_dir:
        print(f"\n正在保存评估结果到: {args.save_dir}")
        
        # 生成报告
        generate_report(metrics, args.save_dir)
        
        # 绘制混淆矩阵
        cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_preds, all_targets, class_names, save_path=cm_path)
    
    # ========== 可视化 ==========
    if args.visualize:
        print(f"\n正在可视化 {args.num_samples} 个样本...")
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        vis_dir = os.path.join(args.save_dir, 'visualizations') if args.save_dir else None
        visualize_predictions(model, dataset, indices, device, save_dir=vis_dir)
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
