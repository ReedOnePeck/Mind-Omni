import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from collections import Counter
import random

#https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb

# 初始化图像处理器和编码器
image_processor = CLIPImageProcessor.from_pretrained(
    '/nfs/diskstation/DataStation/ChangdeDu/clip-vit-large-patch14-336')
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    '/nfs/diskstation/DataStation/ChangdeDu/clip-vit-large-patch14-336')

# 设置为评估模式并禁用梯度
image_encoder.eval()
image_encoder.requires_grad_(False)

# 加载美学预测模型
m = nn.Linear(768, 1)
s = torch.load(
    "/nfs/diskstation/DataStation/ChangdeDu/Muddit/aesthetic-predictor-main/sa_0_4_vit_l_14_linear.pth",
    weights_only=True
)
m.load_state_dict(s)
m.eval()

# 将模型移到GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_encoder.to(device)
m.to(device)


# 定义数据集类
class AestheticDataset(Dataset):
    def __init__(self, image_dir, num_images=73000, sample_size=5000):
        self.image_dir = image_dir
        self.num_images = num_images

        # 随机选择5000个不重复的索引
        self.selected_indices = random.sample(range(num_images), sample_size)

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        image_idx = self.selected_indices[idx]
        image_path = os.path.join(self.image_dir, f"{image_idx:05d}.png")
        return image_path, image_idx


# 自定义collate函数来处理图像加载和预处理
def custom_collate_fn(batch):
    image_paths, indices = zip(*batch)
    images = []

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"无法加载图像 {path}: {e}")
            # 如果无法加载图像，使用一个默认图像或跳过
            # 这里我们简单地使用一个黑色图像作为替代
            images.append(Image.new('RGB', (336, 336), (0, 0, 0)))

    # 使用图像处理器处理所有图像
    processed_images = image_processor(images, return_tensors="pt")

    return processed_images.pixel_values, torch.tensor(indices)


# 创建数据加载器
image_dir = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_imgs"
dataset = AestheticDataset(image_dir)
dataloader = DataLoader(
    dataset,
    batch_size=64,  # 根据GPU内存调整批次大小
    num_workers=4,  # 根据CPU核心数调整工作进程数
    collate_fn=custom_collate_fn,  # 使用自定义的collate函数
    pin_memory=True if torch.cuda.is_available() else False
)

# 收集所有预测值
all_predictions = []

print("开始处理图像...")
with torch.no_grad():
    for batch_pixel_values, batch_indices in dataloader:
        # 将像素值移到设备上
        pixel_values = batch_pixel_values.to(device)

        # 提取特征并计算美学评分
        image_features = image_encoder(pixel_values=pixel_values)
        image_embeds = image_features.image_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        predictions = m(image_embeds)

        # 收集预测值
        all_predictions.extend(predictions.cpu().numpy().flatten().tolist())

        print(f"已处理 {len(all_predictions)} / 5000 张图像")

# 转换为numpy数组以便计算统计量
all_predictions = np.array(all_predictions)

# 计算统计量
mean_value = np.mean(all_predictions)
median_value = np.median(all_predictions)

# 计算众数（可能需要将预测值分组）
# 由于预测值是连续的，我们需要将它们分组到区间中
hist, bin_edges = np.histogram(all_predictions, bins=50)  # 使用50个区间
mode_bin = np.argmax(hist)
mode_value = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2  # 取区间的中点作为众数

print(f"平均值: {mean_value:.4f}")
print(f"中位数: {median_value:.4f}")
print(f"众数: {mode_value:.4f}")
print(f"最小值: {np.min(all_predictions):.4f}")
print(f"最大值: {np.max(all_predictions):.4f}")
print(f"标准差: {np.std(all_predictions):.4f}")


"""
平均值: 6.0135
中位数: 6.0279
众数: 5.8247
最小值: 2.3892
最大值: 9.4005
标准差: 0.8748

"""