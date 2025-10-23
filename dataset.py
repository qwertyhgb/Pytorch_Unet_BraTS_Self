"""
BraTS2021 脑肿瘤分割数据集加载器

功能：
    1. 加载预处理后的 PNG 格式多模态 MRI 图像和分割标签
    2. 支持数据增强（可选）
    3. 自动归一化和类型转换
    4. 提供数据集统计信息

数据格式：
    - 图像：4 通道 PNG (FLAIR, T1, T1CE, T2)，范围 [0, 255]
    - 标签：单通道 PNG，值为 {0, 1, 2, 3}
        0: 背景
        1: 坏死和非增强肿瘤核心 (NCR/NET)
        2: 水肿区域 (ED)
        3: 增强肿瘤 (ET)
"""

import torch
from torch.utils.data import Dataset
import cv2
import glob
import numpy as np
import os
from pathlib import Path


class BraTSDataset(Dataset):
    """
    BraTS2021 数据集类
    
    用于加载预处理后的 2D 切片数据，支持训练和验证
    
    Args:
        img_dir (str): 图像目录路径（包含 imgs 和 masks 子目录）
                      例如：'./data_png' 或 './data_png/imgs'
        transform (callable, optional): 数据增强函数（接收 img, mask 并返回增强后的结果）
                                       例如：albumentations 的 Compose 对象
        validate_data (bool): 是否在初始化时验证数据完整性（默认 True）
    
    数据结构：
        img_dir/
        ├── imgs/
        │   ├── BraTS2021_00000_000.png  (4 通道图像)
        │   ├── BraTS2021_00000_001.png
        │   └── ...
        └── masks/
            ├── BraTS2021_00000_000.png  (单通道标签)
            ├── BraTS2021_00000_001.png
            └── ...
    
    返回格式：
        img (Tensor): [4, H, W]，float32，范围 [0, 1]
        mask (Tensor): [H, W]，int64，值为 {0, 1, 2, 3}
    """
    
    def __init__(self, img_dir, transform=None, validate_data=True):
        """
        初始化数据集
        
        自动检测目录结构：
            - 如果 img_dir 包含 'imgs' 子目录，则使用 imgs/ 和 masks/
            - 否则直接在 img_dir 中查找图像文件
        """
        self.transform = transform
        
        # 智能检测目录结构
        img_dir_path = Path(img_dir)
        imgs_subdir = img_dir_path / "imgs"
        masks_subdir = img_dir_path / "masks"
        
        if imgs_subdir.exists() and masks_subdir.exists():
            # 使用子目录结构
            self.img_dir = str(imgs_subdir)
            self.mask_dir = str(masks_subdir)
            search_pattern = "*.png"
        else:
            # 直接在根目录查找（兼容旧版命名方式）
            self.img_dir = str(img_dir_path)
            self.mask_dir = str(img_dir_path)
            search_pattern = "*_img.png"
        
        # 获取所有图像路径（按文件名排序，确保可重复性）
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, search_pattern)))
        
        # 生成对应的标签路径
        if search_pattern == "*.png":
            # 新版命名：imgs/xxx.png → masks/xxx.png
            self.mask_paths = [
                os.path.join(self.mask_dir, os.path.basename(p))
                for p in self.img_paths
            ]
        else:
            # 旧版命名：xxx_img.png → xxx_mask.png
            self.mask_paths = [
                p.replace("_img.png", "_mask.png")
                for p in self.img_paths
            ]
        
        # 数据验证
        if len(self.img_paths) == 0:
            raise ValueError(
                f"在 {self.img_dir} 中未找到任何图像文件！\n"
                f"请检查：\n"
                f"  1. 目录路径是否正确\n"
                f"  2. 是否已运行数据预处理脚本 (convert_brats_to_png.py)\n"
                f"  3. 图像文件是否存在"
            )
        
        if validate_data:
            self._validate_dataset()
        
        # 打印数据集信息
        print(f"✓ BraTS 数据集加载成功")
        print(f"  图像目录: {self.img_dir}")
        print(f"  标签目录: {self.mask_dir}")
        print(f"  样本数量: {len(self.img_paths)}")
    
    def _validate_dataset(self):
        """
        验证数据集完整性
        
        检查：
            1. 每个图像是否有对应的标签
            2. 文件是否可读
            3. 标签值是否在有效范围内
        """
        missing_masks = []
        invalid_files = []
        
        # 快速检查前 10 个样本（避免初始化过慢）
        check_samples = min(10, len(self.img_paths))
        
        for i in range(check_samples):
            img_path = self.img_paths[i]
            mask_path = self.mask_paths[i]
            
            # 检查标签文件是否存在
            if not os.path.exists(mask_path):
                missing_masks.append(os.path.basename(mask_path))
                continue
            
            # 尝试读取文件
            try:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    invalid_files.append(f"无法读取图像: {os.path.basename(img_path)}")
                elif img.shape[2] != 4:
                    invalid_files.append(f"图像通道数错误 ({img.shape[2]} != 4): {os.path.basename(img_path)}")
                
                if mask is None:
                    invalid_files.append(f"无法读取标签: {os.path.basename(mask_path)}")
                elif mask.max() > 3:
                    invalid_files.append(f"标签值超出范围 (max={mask.max()}): {os.path.basename(mask_path)}")
                    
            except Exception as e:
                invalid_files.append(f"读取错误 {os.path.basename(img_path)}: {str(e)}")
        
        # 报告验证结果
        if missing_masks:
            print(f"⚠ 警告: 发现 {len(missing_masks)} 个缺失的标签文件（前 {check_samples} 个样本）")
            print(f"  示例: {missing_masks[:3]}")
        
        if invalid_files:
            print(f"⚠ 警告: 发现 {len(invalid_files)} 个无效文件（前 {check_samples} 个样本）")
            for err in invalid_files[:3]:
                print(f"  - {err}")
    
    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 样本总数
        """
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
        
        Returns:
            tuple: (img, mask)
                - img (Tensor): [4, H, W]，float32，范围 [0, 1]
                - mask (Tensor): [H, W]，int64，值为 {0, 1, 2, 3}
        
        处理流程：
            1. 读取 4 通道图像和单通道标签
            2. 应用数据增强（如果提供）
            3. 图像归一化到 [0, 1]
            4. 转换为 PyTorch Tensor
            5. 调整维度顺序：[H, W, C] → [C, H, W]
        """
        # 读取图像和标签
        # cv2.IMREAD_UNCHANGED 保留原始通道数和数据类型
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_UNCHANGED)  # [H, W, 4]
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)  # [H, W]
        
        # 错误处理：文件读取失败
        if img is None:
            raise IOError(f"无法读取图像: {self.img_paths[idx]}")
        if mask is None:
            raise IOError(f"无法读取标签: {self.mask_paths[idx]}")
        
        # 数据增强（可选）
        if self.transform is not None:
            # 注意：不同的增强库接口不同
            # albumentations: augmented = transform(image=img, mask=mask)
            # torchvision: img, mask = transform(img, mask)
            try:
                # 尝试 albumentations 风格
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            except (TypeError, KeyError):
                # 回退到函数式调用
                img, mask = self.transform(img, mask)
        
        # 图像预处理
        # 1. 转换维度顺序：[H, W, C] → [C, H, W]（PyTorch 标准格式）
        img = img.transpose(2, 0, 1).astype(np.float32)
        
        # 2. 归一化到 [0, 1]（假设输入范围是 [0, 255]）
        img = img / 255.0
        
        # 标签预处理
        # 转换为 int64（CrossEntropyLoss 要求）
        mask = mask.astype(np.int64)
        
        # 转换为 PyTorch Tensor
        img = torch.from_numpy(img)  # [4, H, W]
        mask = torch.from_numpy(mask)  # [H, W]
        
        return img, mask
    
    def get_sample_info(self, idx):
        """
        获取样本的详细信息（用于调试和可视化）
        
        Args:
            idx (int): 样本索引
        
        Returns:
            dict: 包含文件路径、形状、统计信息等
        """
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # 统计标签分布
        unique, counts = np.unique(mask, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))
        
        info = {
            'index': idx,
            'img_path': img_path,
            'mask_path': mask_path,
            'img_shape': img.shape,
            'mask_shape': mask.shape,
            'img_dtype': img.dtype,
            'mask_dtype': mask.dtype,
            'img_range': (img.min(), img.max()),
            'mask_range': (mask.min(), mask.max()),
            'label_distribution': label_dist,
            'tumor_ratio': (mask > 0).sum() / mask.size,  # 肿瘤区域占比
        }
        
        return info
    
    def get_statistics(self):
        """
        计算数据集的全局统计信息
        
        Returns:
            dict: 包含均值、标准差、类别分布等统计信息
        
        注意：此方法会遍历整个数据集，可能耗时较长
        """
        print("正在计算数据集统计信息（可能需要几分钟）...")
        
        # 初始化统计变量
        pixel_sum = np.zeros(4, dtype=np.float64)
        pixel_sq_sum = np.zeros(4, dtype=np.float64)
        total_pixels = 0
        
        label_counts = np.zeros(4, dtype=np.int64)
        
        # 遍历所有样本
        for idx in range(len(self)):
            img = cv2.imread(self.img_paths[idx], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
            
            # 图像统计（每个通道）
            for c in range(4):
                channel = img[:, :, c]
                pixel_sum[c] += channel.sum()
                pixel_sq_sum[c] += (channel ** 2).sum()
            
            total_pixels += img.shape[0] * img.shape[1]
            
            # 标签统计
            unique, counts = np.unique(mask, return_counts=True)
            for label, count in zip(unique, counts):
                if label < 4:
                    label_counts[label] += count
        
        # 计算均值和标准差
        mean = pixel_sum / total_pixels
        std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)
        
        # 计算类别权重（用于处理类别不平衡）
        class_weights = total_pixels * 4 / (4 * label_counts + 1e-8)
        
        stats = {
            'num_samples': len(self),
            'total_pixels': total_pixels,
            'mean': mean.tolist(),  # 每个通道的均值
            'std': std.tolist(),    # 每个通道的标准差
            'label_counts': label_counts.tolist(),
            'label_distribution': (label_counts / label_counts.sum()).tolist(),
            'class_weights': class_weights.tolist(),
        }
        
        print("✓ 统计完成")
        print(f"  样本数: {stats['num_samples']}")
        print(f"  图像均值: {[f'{m:.4f}' for m in stats['mean']]}")
        print(f"  图像标准差: {[f'{s:.4f}' for s in stats['std']]}")
        print(f"  标签分布: {[f'{d:.2%}' for d in stats['label_distribution']]}")
        
        return stats


# ==================== 数据增强模块 ====================

def get_training_augmentation(image_size=(240, 240)):
    """
    获取训练时的数据增强策略
    
    使用 albumentations 库实现多种数据增强方法，提升模型泛化能力。
    适用于医学图像分割任务，保持图像和标签的空间对应关系。
    
    Args:
        image_size (tuple): 目标图像大小 (H, W)
    
    Returns:
        albumentations.Compose: 数据增强组合
    
    增强方法说明：
        1. 几何变换：翻转、旋转、缩放、平移
        2. 弹性变形：模拟组织形变
        3. 光学变换：亮度、对比度、伽马调整
        4. 噪声添加：高斯噪声、模糊
    
    注意：
        - 所有变换同时应用于图像和标签
        - 使用 border_mode=cv2.BORDER_CONSTANT 避免边界伪影
        - 插值方法：图像用双线性，标签用最近邻
        - CLAHE 已移除（不支持 4 通道图像）
    """
    try:
        import albumentations as A
    except ImportError:
        print("⚠ 警告: 未安装 albumentations，请运行: pip install albumentations")
        return None
    
    train_transform = A.Compose([
        # ========== 1. 几何变换 ==========
        
        # 随机水平翻转（概率 50%）
        # 医学图像左右对称，翻转不影响语义
        A.HorizontalFlip(p=0.5),
        
        # 随机垂直翻转（概率 50%）
        A.VerticalFlip(p=0.5),
        
        # 随机旋转（±15 度）
        # 模拟不同的扫描角度
        A.Rotate(
            limit=15,  # 旋转角度范围 [-15, 15]
            interpolation=cv2.INTER_LINEAR,  # 图像使用双线性插值
            border_mode=cv2.BORDER_CONSTANT,  # 边界填充常数（黑色）
            value=0,  # 图像填充值
            mask_value=0,  # 标签填充值（背景类）
            p=0.5
        ),
        
        # 随机缩放和平移
        # 模拟不同的视野范围
        A.ShiftScaleRotate(
            shift_limit=0.1,   # 平移范围：图像尺寸的 ±10%
            scale_limit=0.1,   # 缩放范围：±10%
            rotate_limit=0,    # 旋转已在上面处理
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5
        ),
        
        # ========== 2. 弹性变形 ==========
        
        # 弹性变形（Elastic Transform）
        # 模拟组织的非刚性形变，对医学图像特别有效
        A.ElasticTransform(
            alpha=50,          # 变形强度
            sigma=5,           # 高斯核标准差（控制变形平滑度）
            alpha_affine=5,    # 仿射变换强度
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3  # 较低概率，避免过度变形
        ),
        
        # 网格扭曲（Grid Distortion）
        # 另一种形变方式，模拟局部扭曲
        A.GridDistortion(
            num_steps=5,       # 网格步数
            distort_limit=0.3, # 扭曲程度
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3
        ),
        
        # ========== 3. 光学变换 ==========
        
        # 随机亮度和对比度调整
        # 模拟不同的扫描参数和设备差异
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # 亮度调整范围 ±20%
            contrast_limit=0.2,    # 对比度调整范围 ±20%
            p=0.5
        ),
        
        # 随机伽马变换
        # 调整图像的非线性亮度响应
        A.RandomGamma(
            gamma_limit=(80, 120),  # 伽马值范围 [0.8, 1.2]
            p=0.5
        ),
        
        # CLAHE（对比度受限自适应直方图均衡）
        # 注意：CLAHE 只支持 1/3 通道，BraTS 是 4 通道，所以跳过
        # 如需使用，需要对每个通道单独应用
        # A.CLAHE(
        #     clip_limit=2.0,
        #     tile_grid_size=(8, 8),
        #     p=0.3
        # ),
        
        # ========== 4. 噪声和模糊 ==========
        
        # 高斯噪声
        # 模拟图像采集过程中的噪声
        A.GaussNoise(
            var_limit=(10.0, 50.0),  # 噪声方差范围
            mean=0,                   # 噪声均值
            p=0.3
        ),
        
        # 高斯模糊
        # 模拟轻微的图像模糊
        A.GaussianBlur(
            blur_limit=(3, 5),  # 模糊核大小（奇数）
            p=0.2
        ),
        
        # ========== 5. 其他增强 ==========
        
        # 随机擦除（Cutout）
        # 随机遮挡部分区域，提高模型鲁棒性
        A.CoarseDropout(
            max_holes=8,           # 最多擦除 8 个区域
            max_height=20,         # 每个区域最大高度
            max_width=20,          # 每个区域最大宽度
            min_holes=1,           # 至少擦除 1 个区域
            min_height=10,
            min_width=10,
            fill_value=0,          # 填充值（黑色）
            mask_fill_value=0,     # 标签填充值（背景）
            p=0.2
        ),
        
    ], additional_targets={'mask': 'mask'})  # 指定标签的处理方式
    
    return train_transform


def get_validation_augmentation(image_size=(240, 240)):
    """
    获取验证/测试时的数据增强策略
    
    验证时通常不使用数据增强，或只使用必要的预处理
    
    Args:
        image_size (tuple): 目标图像大小
    
    Returns:
        albumentations.Compose: 数据增强组合（通常为空或仅包含归一化）
    """
    try:
        import albumentations as A
    except ImportError:
        return None
    
    # 验证时不使用数据增强，保持原始数据
    # 如果需要调整大小，可以添加 A.Resize()
    val_transform = A.Compose([
        # 可选：调整图像大小
        # A.Resize(height=image_size[0], width=image_size[1], 
        #          interpolation=cv2.INTER_LINEAR, p=1.0),
    ], additional_targets={'mask': 'mask'})
    
    return val_transform


def get_light_augmentation():
    """
    获取轻量级数据增强策略
    
    适用于数据量较大或训练时间有限的情况
    只包含最基本和最有效的增强方法
    
    Returns:
        albumentations.Compose: 轻量级数据增强组合
    """
    try:
        import albumentations as A
    except ImportError:
        return None
    
    light_transform = A.Compose([
        # 基础几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
        
        # 基础光学变换
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
    ], additional_targets={'mask': 'mask'})
    
    return light_transform


def get_heavy_augmentation():
    """
    获取重度数据增强策略
    
    适用于数据量较小的情况，使用更激进的增强方法
    注意：过度增强可能导致模型性能下降
    
    Returns:
        albumentations.Compose: 重度数据增强组合
    """
    try:
        import albumentations as A
    except ImportError:
        return None
    
    heavy_transform = A.Compose([
        # 更激进的几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, 
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.7),
        
        # 更强的弹性变形
        A.ElasticTransform(alpha=100, sigma=10, alpha_affine=10, 
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.5, 
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
        
        # 更强的光学变换
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.7),
        # CLAHE 不支持 4 通道图像，已移除
        # A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        
        # 更多噪声
        A.GaussNoise(var_limit=(10.0, 100.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # 更大的擦除
        A.CoarseDropout(max_holes=16, max_height=30, max_width=30, 
                       min_holes=1, min_height=10, min_width=10,
                       fill_value=0, mask_fill_value=0, p=0.3),
        
    ], additional_targets={'mask': 'mask'})
    
    return heavy_transform


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """
    测试代码：验证数据集加载和基本功能
    """
    print("=" * 60)
    print("BraTS 数据集测试")
    print("=" * 60)
    
    # 创建数据集实例
    try:
        dataset = BraTSDataset("./data_png", validate_data=True)
    except Exception as e:
        print(f"✗ 数据集加载失败: {str(e)}")
        exit(1)
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试单个样本
    if len(dataset) > 0:
        print("\n测试第一个样本:")
        img, mask = dataset[0]
        print(f"  图像形状: {img.shape} (应为 [4, H, W])")
        print(f"  图像类型: {img.dtype} (应为 torch.float32)")
        print(f"  图像范围: [{img.min():.3f}, {img.max():.3f}] (应在 [0, 1])")
        print(f"  标签形状: {mask.shape} (应为 [H, W])")
        print(f"  标签类型: {mask.dtype} (应为 torch.int64)")
        print(f"  标签范围: [{mask.min()}, {mask.max()}] (应在 [0, 3])")
        print(f"  标签唯一值: {torch.unique(mask).tolist()}")
        
        # 获取详细信息
        print("\n样本详细信息:")
        info = dataset.get_sample_info(0)
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # 测试 DataLoader
    print("\n测试 DataLoader:")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch_img, batch_mask = next(iter(loader))
    
    print(f"  批次图像形状: {batch_img.shape} (应为 [B, 4, H, W])")
    print(f"  批次标签形状: {batch_mask.shape} (应为 [B, H, W])")
    
    # 计算统计信息（可选，耗时较长）
    # print("\n计算数据集统计信息:")
    # stats = dataset.get_statistics()
    
    # ========== 测试数据增强 ==========
    print("\n" + "=" * 60)
    print("测试数据增强")
    print("=" * 60)
    
    try:
        import albumentations as A
        
        # 测试训练增强
        print("\n1. 测试训练数据增强")
        train_aug = get_training_augmentation()
        if train_aug is not None:
            print(f"  ✓ 训练增强包含 {len(train_aug.transforms)} 种变换")
            
            # 应用增强到第一个样本
            if len(dataset) > 0:
                img_np = cv2.imread(dataset.img_paths[0], cv2.IMREAD_UNCHANGED)
                mask_np = cv2.imread(dataset.mask_paths[0], cv2.IMREAD_UNCHANGED)
                
                augmented = train_aug(image=img_np, mask=mask_np)
                aug_img = augmented['image']
                aug_mask = augmented['mask']
                
                print(f"  原始图像形状: {img_np.shape}")
                print(f"  增强后图像形状: {aug_img.shape}")
                print(f"  原始标签唯一值: {np.unique(mask_np)}")
                print(f"  增强后标签唯一值: {np.unique(aug_mask)}")
        
        # 测试轻量级增强
        print("\n2. 测试轻量级数据增强")
        light_aug = get_light_augmentation()
        if light_aug is not None:
            print(f"  ✓ 轻量级增强包含 {len(light_aug.transforms)} 种变换")
        
        # 测试重度增强
        print("\n3. 测试重度数据增强")
        heavy_aug = get_heavy_augmentation()
        if heavy_aug is not None:
            print(f"  ✓ 重度增强包含 {len(heavy_aug.transforms)} 种变换")
        
        # 创建带增强的数据集
        print("\n4. 测试带增强的数据集")
        aug_dataset = BraTSDataset("./data_png", transform=train_aug, validate_data=False)
        aug_img, aug_mask = aug_dataset[0]
        print(f"  增强数据集样本形状: img={aug_img.shape}, mask={aug_mask.shape}")
        print(f"  增强数据集标签范围: [{aug_mask.min()}, {aug_mask.max()}]")
        
    except ImportError:
        print("\n⚠ 未安装 albumentations，跳过数据增强测试")
        print("  安装命令: pip install albumentations")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    
    # ========== 可视化数据增强效果（可选）==========
    print("\n提示：如需可视化数据增强效果，可以运行以下代码：")
    print("""
    import matplotlib.pyplot as plt
    from dataset import BraTSDataset, get_training_augmentation
    
    # 加载数据集
    dataset = BraTSDataset("./data_png", validate_data=False)
    train_aug = get_training_augmentation()
    
    # 获取一个样本
    img_np = cv2.imread(dataset.img_paths[0], cv2.IMREAD_UNCHANGED)
    mask_np = cv2.imread(dataset.mask_paths[0], cv2.IMREAD_UNCHANGED)
    
    # 应用多次增强，查看效果
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        augmented = train_aug(image=img_np, mask=mask_np)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        axes[0, i].imshow(aug_img[:, :, 0], cmap='gray')
        axes[0, i].set_title(f'增强图像 {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(aug_mask, cmap='jet', vmin=0, vmax=3)
        axes[1, i].set_title(f'增强标签 {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    """)
