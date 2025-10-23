"""
U-Net 2D 医学图像分割模型
用于 BraTS2021 脑肿瘤多模态 MRI 分割任务

模型架构：
    - 编码器（下采样）：4 层，通道数逐渐增加 (64→128→256→512)
    - 解码器（上采样）：3 层，通道数逐渐减少，使用跳跃连接融合特征
    - 输出层：1x1 卷积映射到目标类别数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    双卷积模块：U-Net 的基本构建块
    
    结构：Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU
    
    Args:
        in_ch (int): 输入通道数
        out_ch (int): 输出通道数
        mid_ch (int, optional): 中间层通道数，默认为 None（使用 out_ch）
    """
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        
        self.conv = nn.Sequential(
            # 第一次卷积：3x3 卷积核，padding=1 保持尺寸不变
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),  # 批归一化，加速训练并提高稳定性
            nn.ReLU(inplace=True),   # ReLU 激活函数，inplace 节省内存
            
            # 第二次卷积：进一步提取特征
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入特征图 [B, C, H, W]
        
        Returns:
            Tensor: 输出特征图 [B, out_ch, H, W]
        """
        return self.conv(x)


class Down(nn.Module):
    """
    下采样模块：MaxPooling + DoubleConv
    
    用于编码器路径，逐步降低空间分辨率并增加通道数
    
    Args:
        in_ch (int): 输入通道数
        out_ch (int): 输出通道数
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2 最大池化，尺寸减半
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入特征图 [B, C, H, W]
        
        Returns:
            Tensor: 输出特征图 [B, out_ch, H/2, W/2]
        """
        return self.pool_conv(x)


class Up(nn.Module):
    """
    上采样模块：ConvTranspose2d（或双线性插值）+ DoubleConv
    
    用于解码器路径，逐步恢复空间分辨率并融合编码器特征
    
    Args:
        in_ch (int): 输入通道数
        out_ch (int): 输出通道数
        bilinear (bool): 是否使用双线性插值上采样（默认使用转置卷积）
    """
    def __init__(self, in_ch, out_ch, bilinear=False):
        super().__init__()
        
        # 选择上采样方式
        if bilinear:
            # 双线性插值：参数量少，速度快，但效果可能略差
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            # 转置卷积：可学习参数，效果通常更好
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """
        前向传播：上采样 + 跳跃连接 + 卷积
        
        Args:
            x1 (Tensor): 来自解码器的低分辨率特征图 [B, C1, H, W]
            x2 (Tensor): 来自编码器的高分辨率特征图（跳跃连接）[B, C2, H*2, W*2]
        
        Returns:
            Tensor: 融合后的特征图 [B, out_ch, H*2, W*2]
        """
        # 上采样 x1
        x1 = self.up(x1)
        
        # 处理尺寸不匹配的情况（由于池化可能导致的尺寸差异）
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        # 对 x1 进行填充，使其与 x2 尺寸一致
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # 跳跃连接：在通道维度拼接编码器和解码器特征
        x = torch.cat([x2, x1], dim=1)
        
        # 双卷积融合特征
        return self.conv(x)


class UNet2D(nn.Module):
    """
    U-Net 2D 分割网络
    
    经典的 U 型编码器-解码器架构，适用于医学图像分割任务
    
    Args:
        in_ch (int): 输入通道数（BraTS2021 为 4，对应 4 种 MRI 模态）
        out_ch (int): 输出通道数（分割类别数，BraTS2021 为 4：背景 + 3 类肿瘤）
        bilinear (bool): 是否使用双线性插值上采样（默认 False，使用转置卷积）
        base_ch (int): 基础通道数（默认 64）
    
    网络结构：
        输入 [B, 4, 240, 240]
        ↓ down1 → [B, 64, 240, 240]
        ↓ down2 → [B, 128, 120, 120]
        ↓ down3 → [B, 256, 60, 60]
        ↓ down4 → [B, 512, 30, 30]  (瓶颈层)
        ↑ up1   → [B, 256, 60, 60]  (+ skip connection from down3)
        ↑ up2   → [B, 128, 120, 120] (+ skip connection from down2)
        ↑ up3   → [B, 64, 240, 240]  (+ skip connection from down1)
        输出 [B, 4, 240, 240]
    """
    def __init__(self, in_ch=4, out_ch=4, bilinear=False, base_ch=64):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bilinear = bilinear
        
        # 编码器（下采样路径）
        self.inc = DoubleConv(in_ch, base_ch)  # 初始卷积，不改变尺寸
        self.down1 = Down(base_ch, base_ch * 2)      # 64 → 128
        self.down2 = Down(base_ch * 2, base_ch * 4)  # 128 → 256
        self.down3 = Down(base_ch * 4, base_ch * 8)  # 256 → 512
        
        # 瓶颈层（最深层，特征最抽象）
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch * 8, base_ch * 16 // factor)  # 512 → 1024 (或 512)
        
        # 解码器（上采样路径）
        self.up1 = Up(base_ch * 16, base_ch * 8 // factor, bilinear)  # 1024 → 512
        self.up2 = Up(base_ch * 8, base_ch * 4 // factor, bilinear)   # 512 → 256
        self.up3 = Up(base_ch * 4, base_ch * 2 // factor, bilinear)   # 256 → 128
        self.up4 = Up(base_ch * 2, base_ch, bilinear)                 # 128 → 64
        
        # 输出层：1x1 卷积映射到目标类别数
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入图像 [B, in_ch, H, W]
                       例如：[8, 4, 240, 240] (batch_size=8, 4 个 MRI 模态)
        
        Returns:
            Tensor: 分割预测 logits [B, out_ch, H, W]
                   例如：[8, 4, 240, 240] (4 个类别的未归一化分数)
        """
        # 编码器：逐层下采样并保存特征图（用于跳跃连接）
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024, H/16, W/16] (瓶颈层)
        
        # 解码器：逐层上采样并融合编码器特征
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]
        
        # 输出层：映射到类别数
        logits = self.outc(x)  # [B, out_ch, H, W]
        
        return logits

    def use_checkpointing(self):
        """
        启用梯度检查点（Gradient Checkpointing）
        
        用于减少显存占用，适合大模型或大 batch size 训练
        会略微增加训练时间（约 20%），但可以显著降低显存使用
        """
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)


if __name__ == "__main__":
    """
    模型测试代码
    """
    # 创建模型
    model = UNet2D(in_ch=4, out_ch=4, bilinear=False)
    
    # 打印模型结构
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 4, 240, 240)  # [batch_size, channels, height, width]
    with torch.no_grad():
        output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
