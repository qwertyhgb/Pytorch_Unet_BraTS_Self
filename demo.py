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