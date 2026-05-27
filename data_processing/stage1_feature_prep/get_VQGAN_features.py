import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from PIL import Image
from diffusers import VQModel
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端

# 设置设备
device = 'cuda:5'


def process_image(image, size, Norm=False):
    # 处理 EXIF 方向信息
    try:
        image = exif_transpose(image)
    except:
        pass  # 如果 exif_transpose 不可用，跳过

    if not image.mode == "RGB":
        image = image.convert("RGB")

    # 先调整大小
    image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)(image)

    # 使用中心裁剪而不是随机裁剪，确保结果一致性
    image = transforms.CenterCrop(size)(image)
    image = transforms.ToTensor()(image)

    if Norm:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

    return image


class ImageCaptionLargeDataset(Dataset):
    def __init__(
            self,
            root_dir,
            size=512,
            norm=False
    ):
        self.root_dir = root_dir
        self.size = size
        self.norm = norm

        # 创建包含所有图像文件名的列表
        self.data_list = []

        # 生成从 00000 到 72999 的文件名
        for i in range(73000):
            filename = f"{str(i).zfill(5)}.png"
            file_path = os.path.join(root_dir, filename)

            # 检查文件是否存在
            if os.path.exists(file_path):
                self.data_list.append(filename)
            else:
                print(f"警告: 文件 {file_path} 不存在")

        print(f"找到 {len(self.data_list)} 个有效图像文件")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 获取文件名
        filename = self.data_list[idx]
        img_path = os.path.join(self.root_dir, filename)

        image = Image.open(img_path).convert("RGB")
        processed_image = process_image(image, self.size, self.norm)
        return processed_image, filename  # 同时返回文件名以便后续保存


# 加载VQVAE模型
vq_model = VQModel.from_pretrained("/data/home/luyizhuo/Datastation_lyz/Models/Muddit", subfolder="vqvae")
vq_model.requires_grad_(False)
vq_model.to(device)
vq_model.eval()
vae_scale_factor = 2 ** (len(vq_model.config.block_out_channels) - 1)
print("VQVAE模型加载完成")

# 创建数据集和数据加载器
root_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/"
dataset = ImageCaptionLargeDataset(root_dir, size=512, norm=False)
batch_size = 64  # 根据GPU内存调整批次大小
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 创建输出目录
output_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/VQVAE_continuos_feature"
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# 处理所有图像
height = 512
width = 512

with torch.no_grad():
    for batch_idx, (batch_images, batch_filenames) in enumerate(tqdm(dataloader, desc="处理图像批次")):
        batch_images = batch_images.to(device)

        # 使用VQVAE编码图像
        try:
            # 编码图像
            encoded = vq_model.encode(batch_images).latents.cpu().numpy()


            # 保存每个图像的VQ编码
            for i in range(len(batch_filenames)):
                # 从图像文件名生成对应的npy文件名
                base_name = os.path.splitext(batch_filenames[i])[0]
                output_path = os.path.join(output_dir, f"{base_name}.npy")

                # 保存VQ编码
                np.save(output_path, encoded[i])

        except Exception as e:
            print(f"处理批次 {batch_idx} 时出错: {e}")
            # 继续处理下一个批次
            continue

print("所有图像处理完成!")