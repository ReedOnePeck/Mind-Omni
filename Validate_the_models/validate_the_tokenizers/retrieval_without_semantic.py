import os
import sys
from dataclasses import dataclass
import json
from safetensors.torch import load_file
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
import torch
import PIL
import numpy as np
from MindOmni_utils.trainer_utils import load_images_to_tensor
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from Validate_the_models.validate_the_tokenizers.fMRI_tokenizer_mean import VQ_fMRI
from diffusers.utils import BaseOutput

device = 'cuda:5'


class NSDMultiSubjectDataset(Dataset):
    def __init__(self, subject_ids=[1, 2, 5, 7], base_path=None):
        """
        NSD多被试数据集

        Args:
            subject_ids: 被试ID列表，默认为[1, 2, 5, 7]
            base_path: 数据基础路径
        """
        self.subject_ids = subject_ids
        self.base_path = base_path or "/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_multi"

        # 加载fMRI数据
        self.fmri_data = {}
        for sub_id in subject_ids:
            fmri_path = f"{self.base_path}/test_data_sub{sub_id}/sub{sub_id}_test_multi.npy"
            self.fmri_data[f'sub{sub_id}'] = np.load(fmri_path)

        # 加载图像和文本特征
        self.img_features = np.load(
            '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy')
        self.text_features = np.load(
            '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_feature_1024/text/text_CLIP_H_feature_1024.npy')

        # 加载共享的索引
        index_path = f"{self.base_path}/test_data_sub2/test_img_index_start_from0.npy"
        self.indices = np.load(index_path)

        # 验证数据形状
        self.num_samples = len(self.indices)
        print(f"数据集包含 {self.num_samples} 个样本")

        for sub_id in subject_ids:
            assert self.fmri_data[f'sub{sub_id}'].shape[0] == self.num_samples, \
                f"被试{sub_id}的fMRI数据样本数不匹配"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取当前样本的索引
        feature_idx = self.indices[idx]

        # 获取所有被试的fMRI数据
        fmri_samples = {}
        for sub_id in self.subject_ids:
            fmri_samples[f'sub{sub_id}_fmri'] = torch.FloatTensor(
                self.fmri_data[f'sub{sub_id}'][idx]
            )

        # 获取对应的图像和文本特征
        img_feature = torch.FloatTensor(self.img_features[feature_idx])
        text_feature = torch.FloatTensor(self.text_features[feature_idx])

        # 返回所有数据
        return {
            **fmri_samples,
            'img_feature': img_feature,
            'text_feature': text_feature,
            'feature_index': feature_idx,
            'sample_index': idx
        }


import torch
import torch.nn.functional as F


def calculate_retrieval(model,
                        fmri_data: torch.Tensor,
                        img_clip_feature: torch.Tensor,
                        text_clip_feature: torch.Tensor,
                        top_k_list: list = [50]) -> dict:
    """
    计算fMRI到图像和文本的跨模态检索准确率

    Args:
        model: 模型实例，包含encoder, quant_conv, quantize, fmri_to_clip_proj等组件
        fmri_data (torch.Tensor): 整个测试集的fMRI数据, shape: (N, n_voxel), e.g., (1000, 16127).
        img_clip_feature (torch.Tensor): 对应的图像CLIP特征, shape: (N, clip_dim), e.g., (1000, 1024).
        text_clip_feature (torch.Tensor): 对应的文本CLIP特征, shape: (N, clip_dim), e.g., (1000, 1024).
        top_k_list (list): 要计算的top-k准确率列表，默认为[1, 5, 10]

    Returns:
        dict: 包含检索结果的字典，包含各个top-k的准确率

    示例:
        retrieval_results = calculate_retrieval(
            model=fMRI_quantizer,
            fmri_data=test_fmri_data,
            img_clip_feature=test_img_features,
            text_clip_feature=test_text_features,
            top_k_list=[1, 5, 10]
        )
    """
    model.eval()
    with torch.no_grad():
        # --- 1. 将所有fMRI数据转换为CLIP空间的特征 ---
        h = model.encoder(fmri_data)
        h_pre_quant = model.quant_conv(h)
        quantized_fmri_conv, _, _, _ = model.quantize(h_pre_quant.permute(0, 2, 1))

        fmri_tokens = quantized_fmri_conv.permute(0, 2, 1)
        batch_size = fmri_tokens.shape[0]
        fmri_tokens_flattened = fmri_tokens.reshape(batch_size, -1)  # Shape: (B, 16 * 64) -> (B, 1024)
        fmri_clip_projected = model.fmri_to_clip_proj(fmri_tokens_flattened)

        # --- 2. 特征归一化 (为计算余弦相似度做准备) ---
        fmri_norm = F.normalize(fmri_clip_projected, p=2, dim=-1)
        img_norm = F.normalize(img_clip_feature, p=2, dim=-1)
        text_norm = F.normalize(text_clip_feature, p=2, dim=-1)

        # --- 3. 计算相似度矩阵 ---
        # fMRI到图像相似度矩阵: (N_fmri, N_img)
        sim_matrix_img = torch.matmul(fmri_norm, img_norm.t())
        # fMRI到文本相似度矩阵: (N_fmri, N_text)
        sim_matrix_text = torch.matmul(fmri_norm, text_norm.t())

        # --- 4. 计算各个top-k的准确率 ---
        num_samples = fmri_data.shape[0]
        ground_truth_indices = torch.arange(num_samples, device=fmri_data.device)

        results = {}

        # 对每个top-k值计算准确率
        for top_k in top_k_list:
            # fMRI到图像的top-k准确率
            _, topk_indices_img = torch.topk(sim_matrix_img, k=top_k, dim=1)
            correct_img = torch.any(topk_indices_img == ground_truth_indices.unsqueeze(1), dim=1)
            img_acc = correct_img.float().mean()

            # fMRI到文本的top-k准确率
            _, topk_indices_text = torch.topk(sim_matrix_text, k=top_k, dim=1)
            correct_text = torch.any(topk_indices_text == ground_truth_indices.unsqueeze(1), dim=1)
            text_acc = correct_text.float().mean()

            # 保存结果
            results[f'fmri_to_image_top{top_k}_acc'] = img_acc.item()
            results[f'fmri_to_text_top{top_k}_acc'] = text_acc.item()

        # 为了向后兼容，保留原来的键名（对应top1）
        if 1 in top_k_list:
            results['fmri_to_image_acc'] = results['fmri_to_image_top1_acc']
            results['fmri_to_text_acc'] = results['fmri_to_text_top1_acc']

        return results




brain_vae = VQ_fMRI.from_pretrained(
        "/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/fMRI_tokenizer/train_without_semantic/checkpoint-1000/VQ_fMRI")
brain_vae.requires_grad_(False)
brain_vae = brain_vae.to(device)

# 创建数据集
dataset = NSDMultiSubjectDataset(subject_ids=[1, 2, 5, 7])
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# 获取整个数据集
batch = next(iter(dataloader))

# 提取图像和文本特征（对所有被试都是一样的）
img_features_all = batch['img_feature'].to(device)
text_features_all = batch['text_feature'].to(device)

# 为每个被试计算检索准确率
subject_results = {}

for sub_id in [1, 2, 5, 7]:
    print(f"\n=== 计算被试 {sub_id} 的检索准确率 ===")

    # 获取该被试的fMRI数据
    fmri_key = f'sub{sub_id}_fmri'
    fmri_data = batch[fmri_key].to(device)

    # 计算检索准确率
    results = calculate_retrieval(
        model=brain_vae,
        fmri_data=fmri_data,
        img_clip_feature=img_features_all,
        text_clip_feature=text_features_all,
        top_k_list=[50]  # 只计算top50
    )

    subject_results[sub_id] = results

    # 打印当前被试结果
    print(f"被试 {sub_id} 结果:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

# 计算均值和标准差
fmri2img_accs = []
fmri2text_accs = []

for sub_id, results in subject_results.items():
    fmri2img_accs.append(results['fmri_to_image_top50_acc'])
    fmri2text_accs.append(results['fmri_to_text_top50_acc'])

# 转换为numpy数组方便计算
fmri2img_accs = np.array(fmri2img_accs)
fmri2text_accs = np.array(fmri2text_accs)

# 计算统计量
fmri2img_mean = np.mean(fmri2img_accs)
fmri2img_std = np.std(fmri2img_accs)
fmri2text_mean = np.mean(fmri2text_accs)
fmri2text_std = np.std(fmri2text_accs)

# 打印最终结果
print("\n" + "=" * 60)
print("跨模态检索评估结果汇总 (Top-50)")
print("=" * 60)
print(f"fMRI -> 图像检索准确率:")
print(f"  均值: {fmri2img_mean:.4f} ± {fmri2img_std:.4f}")
print(f"  各被试结果: {fmri2img_accs}")
print(f"fMRI -> 文本检索准确率:")
print(f"  均值: {fmri2text_mean:.4f} ± {fmri2text_std:.4f}")
print(f"  各被试结果: {fmri2text_accs}")

# 计算随机准确率作为参考
num_samples = len(dataset)
chance_level_50 = 50 / num_samples
print(f"\n随机准确率参考 (Top-50): {chance_level_50:.4f}")

# 可选：保存结果到文件
results_summary = {
    'fmri_to_image': {
        'mean': float(fmri2img_mean),
        'std': float(fmri2img_std),
        'subject_results': {f'sub{id}': float(acc) for id, acc in zip([1, 2, 5, 7], fmri2img_accs)}
    },
    'fmri_to_text': {
        'mean': float(fmri2text_mean),
        'std': float(fmri2text_std),
        'subject_results': {f'sub{id}': float(acc) for id, acc in zip([1, 2, 5, 7], fmri2text_accs)}
    },
    'chance_level_top50': float(chance_level_50),
    'num_samples': num_samples
}

# 保存结果
import json

with open('retrieval_results_top50.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n结果已保存到: retrieval_results_top50.json")




