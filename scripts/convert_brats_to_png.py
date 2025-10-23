"""
BraTS2021 数据预处理脚本
将 NIfTI 格式（.nii.gz）的 3D MRI 数据转换为 PNG 格式的 2D 切片

功能：
    1. 读取 BraTS2021 数据集的多模态 MRI 图像（FLAIR, T1, T1CE, T2）
    2. 读取分割标签（seg）
    3. 对每个 2D 切片进行标准化处理
    4. 将 4 个模态合并为 4 通道图像
    5. 重映射标签（4→3）并过滤空白切片
    6. 保存为 PNG 格式，便于后续训练

数据格式：
    输入：./data/BraTS2021/{patient_id}/{patient_id}_{modality}.nii.gz
    输出：./data_png/imgs/{patient_id}_{slice_idx:03d}.png (4 通道图像)
         ./data_png/masks/{patient_id}_{slice_idx:03d}.png (单通道标签)
"""

import os
import argparse
import logging
from datetime import datetime
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path


# 配置日志
def setup_logger(log_dir="./logs", log_level=logging.INFO):
    """
    配置日志系统
    
    Args:
        log_dir (str): 日志文件保存目录
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"convert_brats_{timestamp}.log")
    
    # 创建日志记录器
    logger = logging.getLogger("BraTS_Converter")
    logger.setLevel(log_level)
    
    # 清除已有的处理器（避免重复）
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件处理器（保存到文件）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器（输出到终端）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # 控制台只显示警告及以上级别
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    
    return logger


def norm_slice(img_slice):
    """
    对单个 2D 切片进行标准化处理
    
    步骤：
        1. Z-score 标准化（均值为 0，标准差为 1）
        2. 裁剪到 [-5, 5] 范围（去除极端异常值）
        3. Min-Max 归一化到 [0, 255]
        4. 转换为 uint8 类型
    
    Args:
        img_slice (np.ndarray): 输入切片 [H, W]，float 类型
    
    Returns:
        np.ndarray: 标准化后的切片 [H, W]，uint8 类型，范围 [0, 255]
    """
    # 避免除零错误
    std = img_slice.std()
    if std < 1e-8:
        return np.zeros_like(img_slice, dtype=np.uint8)
    
    # Z-score 标准化
    img_normalized = (img_slice - img_slice.mean()) / (std + 1e-8)
    
    # 裁剪异常值（保留 99.7% 的数据，假设正态分布）
    img_clipped = np.clip(img_normalized, -5, 5)
    
    # Min-Max 归一化到 [0, 255]
    img_min = img_clipped.min()
    img_max = img_clipped.max()
    
    if img_max - img_min < 1e-8:
        return np.zeros_like(img_slice, dtype=np.uint8)
    
    img_scaled = ((img_clipped - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    return img_scaled


def remap_labels(seg):
    """
    重映射分割标签
    
    BraTS2021 原始标签：
        0: 背景
        1: 坏死和非增强肿瘤核心 (NCR/NET)
        2: 水肿区域 (ED)
        4: 增强肿瘤 (ET)
    
    重映射后标签（便于训练）：
        0: 背景
        1: NCR/NET
        2: ED
        3: ET (原标签 4)
    
    Args:
        seg (np.ndarray): 原始分割标签 [H, W, D]
    
    Returns:
        np.ndarray: 重映射后的标签 [H, W, D]
    """
    seg_remapped = seg.copy()
    seg_remapped[seg == 4] = 3  # 将标签 4 映射为 3
    return seg_remapped


def is_valid_slice(mask_slice, min_tumor_pixels=10):
    """
    判断切片是否有效（是否包含足够的肿瘤区域）
    
    Args:
        mask_slice (np.ndarray): 分割标签切片 [H, W]
        min_tumor_pixels (int): 最小肿瘤像素数阈值
    
    Returns:
        bool: True 表示有效切片，False 表示空白切片
    """
    tumor_pixels = np.sum(mask_slice > 0)
    return tumor_pixels >= min_tumor_pixels


def process_patient(patient_id, input_dir, output_dir, min_tumor_pixels=10, logger=None):
    """
    处理单个患者的所有切片
    
    Args:
        patient_id (str): 患者 ID（文件夹名称）
        input_dir (str): 输入数据根目录
        output_dir (str): 输出数据根目录（包含 imgs 和 masks 子目录）
        min_tumor_pixels (int): 最小肿瘤像素数阈值
        logger (logging.Logger): 日志记录器
    
    Returns:
        int: 保存的有效切片数量
    """
    patient_path = os.path.join(input_dir, patient_id)
    
    # 检查患者文件夹是否存在
    if not os.path.exists(patient_path):
        if logger:
            logger.warning(f"患者文件夹不存在: {patient_path}")
        return 0
    
    if logger:
        logger.info(f"开始处理患者: {patient_id}")
    
    try:
        # 加载 4 种 MRI 模态
        if logger:
            logger.debug(f"加载 {patient_id} 的 MRI 数据...")
        
        flair = nib.load(os.path.join(patient_path, f"{patient_id}_flair.nii.gz")).get_fdata()
        t1 = nib.load(os.path.join(patient_path, f"{patient_id}_t1.nii.gz")).get_fdata()
        t1ce = nib.load(os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(patient_path, f"{patient_id}_t2.nii.gz")).get_fdata()
        
        # 加载分割标签
        seg = nib.load(os.path.join(patient_path, f"{patient_id}_seg.nii.gz")).get_fdata()
        
        if logger:
            logger.debug(f"数据形状: {flair.shape}, 数据类型: {flair.dtype}")
        
        # 重映射标签
        seg = remap_labels(seg)
        
    except Exception as e:
        if logger:
            logger.error(f"无法加载患者数据 {patient_id}: {str(e)}", exc_info=True)
        return 0
    
    # 获取切片数量（Z 轴）
    num_slices = flair.shape[2]
    valid_slices = 0
    
    # 创建输出子目录
    imgs_dir = os.path.join(output_dir, "imgs")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # 遍历所有切片
    skipped_slices = 0
    for slice_idx in range(num_slices):
        # 提取当前切片
        flair_slice = flair[:, :, slice_idx]
        t1_slice = t1[:, :, slice_idx]
        t1ce_slice = t1ce[:, :, slice_idx]
        t2_slice = t2[:, :, slice_idx]
        mask_slice = seg[:, :, slice_idx].astype(np.uint8)
        
        # 过滤空白切片
        if not is_valid_slice(mask_slice, min_tumor_pixels):
            skipped_slices += 1
            continue
        
        try:
            # 标准化每个模态的切片
            flair_norm = norm_slice(flair_slice)
            t1_norm = norm_slice(t1_slice)
            t1ce_norm = norm_slice(t1ce_slice)
            t2_norm = norm_slice(t2_slice)
            
            # 合并为 4 通道图像 [H, W, 4]
            img_4ch = np.stack([flair_norm, t1_norm, t1ce_norm, t2_norm], axis=-1)
            
            # 生成文件名（不带后缀标识，统一命名）
            filename = f"{patient_id}_{slice_idx:03d}.png"
            
            img_path = os.path.join(imgs_dir, filename)
            mask_path = os.path.join(masks_dir, filename)
            
            # 保存图像和标签到对应子目录
            cv2.imwrite(img_path, img_4ch)
            cv2.imwrite(mask_path, mask_slice)
            
            valid_slices += 1
            
        except Exception as e:
            if logger:
                logger.error(f"处理切片 {patient_id}_{slice_idx:03d} 时出错: {str(e)}")
            continue
    
    if logger:
        logger.info(f"患者 {patient_id} 处理完成: 保存 {valid_slices}/{num_slices} 个切片 (跳过 {skipped_slices} 个空白切片)")
    
    return valid_slices


def main():
    """
    主函数：批量处理所有患者数据
    """
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="BraTS2021 数据预处理：NIfTI → PNG")
    parser.add_argument("--input_dir", type=str, default="./data/BraTS2021",
                        help="输入数据目录（包含患者文件夹）")
    parser.add_argument("--output_dir", type=str, default="./data_png",
                        help="输出数据目录（将创建 imgs 和 masks 子目录）")
    parser.add_argument("--min_tumor_pixels", type=int, default=10,
                        help="最小肿瘤像素数阈值（过滤空白切片）")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="日志文件保存目录")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 初始化日志系统
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(log_dir=args.log_dir, log_level=log_level)
    
    logger.info("=" * 60)
    logger.info("BraTS2021 数据预处理开始")
    logger.info("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录: {args.output_dir}")
    
    # 获取所有患者 ID
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        print(f"错误: 输入目录不存在 - {args.input_dir}")
        return
    
    all_folders = os.listdir(args.input_dir)
    patients = [p for p in all_folders if "BraTS2021" in p and os.path.isdir(os.path.join(args.input_dir, p))]
    
    if len(patients) == 0:
        logger.warning(f"在 {args.input_dir} 中未找到任何 BraTS2021 患者数据")
        print(f"警告: 在 {args.input_dir} 中未找到任何 BraTS2021 患者数据")
        return
    
    logger.info(f"找到 {len(patients)} 个患者数据")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"最小肿瘤像素数阈值: {args.min_tumor_pixels}")
    
    print(f"找到 {len(patients)} 个患者数据")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最小肿瘤像素数: {args.min_tumor_pixels}")
    print(f"日志文件: {args.log_dir}")
    print("-" * 60)
    
    # 批量处理所有患者
    total_slices = 0
    failed_patients = []
    start_time = datetime.now()
    
    for patient_id in tqdm(patients, desc="处理患者数据"):
        num_slices = process_patient(
            patient_id=patient_id,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_tumor_pixels=args.min_tumor_pixels,
            logger=logger
        )
        
        if num_slices == 0:
            failed_patients.append(patient_id)
        
        total_slices += num_slices
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    # 输出统计信息
    print("-" * 60)
    print(f"处理完成！")
    print(f"总患者数: {len(patients)}")
    print(f"成功处理: {len(patients) - len(failed_patients)}")
    print(f"失败患者: {len(failed_patients)}")
    print(f"总切片数: {total_slices}")
    if len(patients) > len(failed_patients):
        print(f"平均每患者: {total_slices / (len(patients) - len(failed_patients)):.1f} 个切片")
    print(f"处理耗时: {elapsed_time:.2f} 秒")
    print(f"输出目录: {args.output_dir}")
    
    logger.info("=" * 60)
    logger.info("处理完成统计")
    logger.info("=" * 60)
    logger.info(f"总患者数: {len(patients)}")
    logger.info(f"成功处理: {len(patients) - len(failed_patients)}")
    logger.info(f"失败患者数: {len(failed_patients)}")
    if failed_patients:
        logger.warning(f"失败患者列表: {', '.join(failed_patients)}")
    logger.info(f"总切片数: {total_slices}")
    if len(patients) > len(failed_patients):
        logger.info(f"平均每患者: {total_slices / (len(patients) - len(failed_patients)):.1f} 个切片")
    logger.info(f"处理耗时: {elapsed_time:.2f} 秒")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
