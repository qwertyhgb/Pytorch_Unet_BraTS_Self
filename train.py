"""
BraTS2021 脑肿瘤分割训练脚本

功能：
    1. 加载 BraTS2021 数据集并划分训练集/验证集
    2. 使用 U-Net 2D 模型进行训练
    3. 计算损失函数（CrossEntropyLoss）和评估指标（Dice Score）
    4. 支持模型保存、学习率调度、早停等功能
    5. 记录训练日志和可视化

训练配置：
    - 模型：U-Net 2D (4 输入通道 → 4 输出类别)
    - 损失函数：CrossEntropyLoss（可选类别权重）
    - 优化器：Adam (lr=1e-4)
    - 评估指标：Dice Score（针对 3 类肿瘤区域）
    - 训练轮数：30 epochs
    - 批次大小：8
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import BraTSDataset
from models.unet2d import UNet2D  # 修正导入路径
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import json


# ==================== 配置参数 ====================

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有训练配置的参数对象
    """
    parser = argparse.ArgumentParser(description="BraTS2021 U-Net 2D 训练脚本")
    
    # 数据相关
    parser.add_argument("--data_dir", type=str, default="./data_png",
                        help="数据集根目录")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例（默认 0.8）")
    
    # 模型相关
    parser.add_argument("--in_channels", type=int, default=4,
                        help="输入通道数（4 种 MRI 模态）")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="输出类别数（背景 + 3 类肿瘤）")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="U-Net 基础通道数")
    parser.add_argument("--bilinear", action="store_true",
                        help="使用双线性插值上采样（默认使用转置卷积）")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减（L2 正则化）")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="数据加载线程数")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="使用混合精度训练（FP16）加速")
    parser.add_argument("--compile_model", action="store_true",
                        help="使用 torch.compile 编译模型（PyTorch 2.0+）")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="梯度累积步数（模拟更大的 batch size）")
    
    # 损失函数相关
    parser.add_argument("--use_class_weights", action="store_true",
                        help="使用类别权重处理类别不平衡")
    parser.add_argument("--dice_weight", type=float, default=0.0,
                        help="Dice Loss 权重（0 表示只使用 CE Loss）")
    
    # 学习率调度
    parser.add_argument("--use_scheduler", action="store_true",
                        help="使用学习率调度器")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="学习率调度器的耐心值")
    parser.add_argument("--scheduler_factor", type=float, default=0.5,
                        help="学习率衰减因子")
    
    # 早停
    parser.add_argument("--early_stopping", action="store_true",
                        help="启用早停机制")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值（验证集无改善的轮数）")
    
    # 保存和日志
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="模型保存目录")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="日志打印间隔（批次数）")
    parser.add_argument("--save_best_only", action="store_true",
                        help="只保存最佳模型")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--validate_every", type=int, default=1,
                        help="每隔多少轮进行一次验证")
    
    return parser.parse_args()


# ==================== 工具函数 ====================

def set_seed(seed):
    """
    设置随机种子，确保实验可重复
    
    Args:
        seed (int): 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def dice_score(pred, target, num_classes=4, eps=1e-5):
    """
    计算 Dice Score（针对多类别分割）- 优化版本
    
    Dice Score 是医学图像分割中常用的评估指标，衡量预测和真实标签的重叠程度
    公式：Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        pred (Tensor): 模型预测 logits [B, C, H, W]
        target (Tensor): 真实标签 [B, H, W]，值为 {0, 1, 2, 3}
        num_classes (int): 类别数（默认 4）
        eps (float): 平滑项，避免除零
    
    Returns:
        float: 平均 Dice Score（只计算肿瘤类别 1, 2, 3，忽略背景）
    """
    # 将 logits 转换为类别预测
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    # 向量化计算所有类别的 Dice Score
    dice_scores = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float()
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())
    
    return sum(dice_scores) / len(dice_scores)


def dice_loss(pred, target, num_classes=4, eps=1e-5):
    """
    计算 Dice Loss（可与 CrossEntropyLoss 结合使用）
    
    Dice Loss = 1 - Dice Score
    
    Args:
        pred (Tensor): 模型预测 logits [B, C, H, W]
        target (Tensor): 真实标签 [B, H, W]
        num_classes (int): 类别数
        eps (float): 平滑项
    
    Returns:
        Tensor: Dice Loss 标量
    """
    # 将 logits 转换为概率分布
    pred_probs = torch.softmax(pred, dim=1)  # [B, C, H, W]
    
    # 将 target 转换为 one-hot 编码
    target_one_hot = torch.nn.functional.one_hot(target, num_classes)  # [B, H, W, C]
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
    
    dice_loss_value = 0.0
    
    # 遍历每个类别（包括背景）
    for c in range(num_classes):
        pred_c = pred_probs[:, c, :, :]  # [B, H, W]
        target_c = target_one_hot[:, c, :, :]  # [B, H, W]
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_loss_value += (1.0 - dice)
    
    return dice_loss_value / num_classes


def save_checkpoint(model, optimizer, epoch, best_dice, save_path, scheduler=None):
    """
    保存训练检查点
    
    Args:
        model (nn.Module): 模型
        optimizer (Optimizer): 优化器
        epoch (int): 当前轮数
        best_dice (float): 最佳 Dice Score
        save_path (str): 保存路径
        scheduler (optional): 学习率调度器
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"✓ 检查点已保存: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None):
    """
    加载训练检查点
    
    Args:
        model (nn.Module): 模型
        optimizer (Optimizer): 优化器
        checkpoint_path (str): 检查点路径
        scheduler (optional): 学习率调度器
    
    Returns:
        tuple: (start_epoch, best_dice)
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_dice = checkpoint.get('best_dice', 0.0)
    
    print(f"✓ 检查点已加载: {checkpoint_path}")
    print(f"  恢复训练从 Epoch {start_epoch}，最佳 Dice: {best_dice:.4f}")
    
    return start_epoch, best_dice


# ==================== 训练和验证函数 ====================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args, scaler=None):
    """
    训练一个 epoch（优化版本）
    
    Args:
        model (nn.Module): 模型
        train_loader (DataLoader): 训练数据加载器
        criterion (nn.Module): 损失函数
        optimizer (Optimizer): 优化器
        device (torch.device): 设备
        epoch (int): 当前轮数
        args (Namespace): 训练参数
        scaler (GradScaler, optional): 混合精度训练的梯度缩放器
    
    Returns:
        tuple: (平均损失, 平均 Dice Score)
    """
    model.train()
    
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(train_loader)
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [训练]")
    
    optimizer.zero_grad()  # 在循环外初始化
    
    for batch_idx, (images, masks) in enumerate(pbar):
        # 将数据移到设备（non_blocking 加速）
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                ce_loss = criterion(outputs, masks)
                
                if args.dice_weight > 0:
                    d_loss = dice_loss(outputs, masks, num_classes=args.num_classes)
                    loss = ce_loss + args.dice_weight * d_loss
                else:
                    loss = ce_loss
                
                # 梯度累积
                loss = loss / args.gradient_accumulation
            
            scaler.scale(loss).backward()
            
            # 每 N 步更新一次参数
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 标准训练
            outputs = model(images)
            ce_loss = criterion(outputs, masks)
            
            if args.dice_weight > 0:
                d_loss = dice_loss(outputs, masks, num_classes=args.num_classes)
                loss = ce_loss + args.dice_weight * d_loss
            else:
                loss = ce_loss
            
            loss = loss / args.gradient_accumulation
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 计算评估指标
        with torch.no_grad():
            batch_dice = dice_score(outputs, masks, num_classes=args.num_classes)
        
        # 累计统计
        total_loss += loss.item() * args.gradient_accumulation
        total_dice += batch_dice
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item() * args.gradient_accumulation:.4f}',
            'dice': f'{batch_dice:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, args):
    """
    验证模型性能
    
    Args:
        model (nn.Module): 模型
        val_loader (DataLoader): 验证数据加载器
        criterion (nn.Module): 损失函数
        device (torch.device): 设备
        epoch (int): 当前轮数
        args (Namespace): 训练参数
    
    Returns:
        tuple: (平均损失, 平均 Dice Score)
    """
    model.eval()
    
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(val_loader)
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [验证]")
    
    for images, masks in pbar:
        # 将数据移到设备
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        ce_loss = criterion(outputs, masks)
        
        if args.dice_weight > 0:
            d_loss = dice_loss(outputs, masks, num_classes=args.num_classes)
            loss = ce_loss + args.dice_weight * d_loss
        else:
            loss = ce_loss
        
        # 计算评估指标
        batch_dice = dice_score(outputs, masks, num_classes=args.num_classes)
        
        # 累计统计
        total_loss += loss.item()
        total_dice += batch_dice
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{batch_dice:.4f}'
        })
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice


# ==================== 主训练流程 ====================

def main():
    """
    主训练函数
    """
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存训练配置
    config_path = os.path.join(args.save_dir, "train_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    print(f"✓ 训练配置已保存: {config_path}")
    
    # ==================== 加载数据集 ====================
    print("\n" + "=" * 60)
    print("加载数据集")
    print("=" * 60)
    
    # 创建完整数据集
    full_dataset = BraTSDataset(args.data_dir, validate_data=True)
    
    # 划分训练集和验证集
    train_size = int(args.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)  # 确保可重复
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 样本 ({args.train_ratio * 100:.0f}%)")
    print(f"  验证集: {len(val_dataset)} 样本 ({(1 - args.train_ratio) * 100:.0f}%)")
    
    # 创建数据加载器（优化版本）
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.num_workers > 0 else False,  # 保持 worker 进程存活
        prefetch_factor=2 if args.num_workers > 0 else None  # 预取数据
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    print(f"\n数据加载器:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  批次大小: {args.batch_size}")
    
    # ==================== 创建模型 ====================
    print("\n" + "=" * 60)
    print("创建模型")
    print("=" * 60)
    
    model = UNet2D(
        in_ch=args.in_channels,
        out_ch=args.num_classes,
        bilinear=args.bilinear,
        base_ch=args.base_channels
    ).to(device)
    
    # 编译模型（PyTorch 2.0+）
    if args.compile_model:
        try:
            print("正在编译模型...")
            model = torch.compile(model)
            print("✓ 模型编译成功")
        except Exception as e:
            print(f"⚠ 模型编译失败: {e}")
            print("  继续使用未编译的模型")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型: U-Net 2D")
    print(f"  输入通道: {args.in_channels}")
    print(f"  输出类别: {args.num_classes}")
    print(f"  基础通道: {args.base_channels}")
    print(f"  上采样方式: {'双线性插值' if args.bilinear else '转置卷积'}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    if args.mixed_precision:
        print(f"  混合精度: 启用 (FP16)")
    if args.gradient_accumulation > 1:
        print(f"  梯度累积: {args.gradient_accumulation} 步 (等效 batch size: {args.batch_size * args.gradient_accumulation})")
    
    # ==================== 创建损失函数 ====================
    if args.use_class_weights:
        # 计算类别权重（处理类别不平衡）
        print("\n正在计算类别权重...")
        stats = full_dataset.get_statistics()
        class_weights = torch.tensor(stats['class_weights'], dtype=torch.float32).to(device)
        print(f"类别权重: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # ==================== 创建优化器 ====================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # ==================== 创建学习率调度器 ====================
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # 监控 Dice Score（越大越好）
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            verbose=True
        )
        print(f"\n学习率调度器: ReduceLROnPlateau")
        print(f"  耐心值: {args.scheduler_patience}")
        print(f"  衰减因子: {args.scheduler_factor}")
    
    # ==================== 混合精度训练 ====================
    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("\n✓ 混合精度训练已启用")
    
    # ==================== 恢复训练（可选）====================
    start_epoch = 1
    best_dice = 0.0
    patience_counter = 0
    
    if args.resume:
        start_epoch, best_dice = load_checkpoint(model, optimizer, args.resume, scheduler)
    
    # ==================== 训练循环 ====================
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    train_history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")
        
        # 训练阶段
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args, scaler
        )
        
        # 验证阶段（每隔 validate_every 轮进行一次）
        if epoch % args.validate_every == 0:
            val_loss, val_dice = validate(
                model, val_loader, criterion, device, epoch, args
            )
        else:
            val_loss, val_dice = 0.0, 0.0
        
        # 记录历史
        train_history['train_loss'].append(train_loss)
        train_history['train_dice'].append(train_dice)
        train_history['val_loss'].append(val_loss)
        train_history['val_dice'].append(val_dice)
        train_history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 打印统计信息
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch} 总结:")
        print(f"  训练损失: {train_loss:.4f} | 训练 Dice: {train_dice:.4f}")
        if epoch % args.validate_every == 0:
            print(f"  验证损失: {val_loss:.4f} | 验证 Dice: {val_dice:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 60}")
        
        # 学习率调度
        if scheduler is not None and epoch % args.validate_every == 0:
            scheduler.step(val_dice)
        
        # 保存模型
        if epoch % args.validate_every == 0:
            is_best = val_dice > best_dice
            
            if is_best:
                best_dice = val_dice
                patience_counter = 0
                
                # 保存最佳模型
                best_model_path = os.path.join(args.save_dir, "best_model.pth")
                save_checkpoint(model, optimizer, epoch, best_dice, best_model_path, scheduler)
                print(f"🎉 新的最佳模型！Dice Score: {best_dice:.4f}")
            else:
                patience_counter += 1
            
            # 保存最新模型（可选）
            if not args.save_best_only:
                latest_model_path = os.path.join(args.save_dir, "latest_model.pth")
                save_checkpoint(model, optimizer, epoch, best_dice, latest_model_path, scheduler)
        
        # 早停检查
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\n⚠ 早停触发！验证集 Dice Score 已 {args.patience} 轮未改善")
            print(f"最佳 Dice Score: {best_dice:.4f}")
            break
    
    # ==================== 训练完成 ====================
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳验证 Dice Score: {best_dice:.4f}")
    print(f"模型保存目录: {args.save_dir}")
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, "train_history.json")
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=4)
    print(f"✓ 训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
