import sys
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
import torch
from collections import defaultdict
import os
from typing import List
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from PIL import Image, ImageDraw, ImageFont
import glob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)


class Stage1_TrainDataset(Dataset):
    def __init__(self, fMRI_root: str, image_token_root: str, text_token_root: str):
        super().__init__()
        self.fMRI_root = fMRI_root
        self.image_token_root = image_token_root
        self.text_token_root = text_token_root

        print(">>> 正在初始化三模态数据集...")

        # 检查路径是否存在
        if not os.path.isdir(self.fMRI_root):
            raise FileNotFoundError(f"fMRI 根目录未找到: {self.fMRI_root}")
        if not os.path.isdir(self.image_token_root):
            raise FileNotFoundError(f"图像 Token 根目录未找到: {self.image_token_root}")
        if not os.path.isdir(self.text_token_root):
            raise FileNotFoundError(f"文本 Token 根目录未找到: {self.text_token_root}")

        # 扫描 fMRI 文件列表作为数据集的主索引
        print(f"正在扫描 fMRI 文件于: {fMRI_root}...")
        self.fmri_file_list = sorted(glob.glob(os.path.join(self.fMRI_root, "*_trial*.npy")))

        if not self.fmri_file_list:
            raise FileNotFoundError(f"在 {self.fMRI_root} 中未找到任何 fMRI 文件。请检查路径和文件格式。")

        self.data_len = len(self.fmri_file_list)
        print(f"数据集初始化成功。共找到 {self.data_len} 个 fMRI 样本。")

    def __len__(self):
        return self.data_len

    def __getitem__(self, index: int) -> dict:
        """
        根据给定的索引，获取一个数据样本。
        """
        # --- 1. 获取 fMRI 数据 ---
        fmri_file_path = self.fmri_file_list[index]
        fmri_data = torch.from_numpy(np.load(fmri_file_path)).float()

        # --- 2. 提取公共索引 ---
        # 从 fMRI 文件名中提取 5 位数的公共索引字符串 (例如 "00123")
        # 假设文件名格式: .../00123_trial_01.npy -> 提取 "00123"
        try:
            common_idx_str = os.path.basename(fmri_file_path).split('_')[0]
            if not (common_idx_str.isdigit() and len(common_idx_str) == 5):
                raise ValueError(f"从文件名 '{fmri_file_path}' 中提取的索引 '{common_idx_str}' 格式不正确。")
        except (IndexError, ValueError) as e:
            raise ValueError(f"无法从文件名 '{fmri_file_path}' 中解析出有效的 5 位数索引。错误: {e}")

        # --- 3. 根据公共索引加载图像 Token ---
        image_token_path = os.path.join(self.image_token_root, f"{common_idx_str}.npy")
        try:
            # 图像 Token IDs 应为整数类型
            img_ids = torch.from_numpy(np.load(image_token_path)).long()
        except FileNotFoundError:
            print(f"警告: 图像 Token 文件未找到: {image_token_path}。将返回一个空张量。")
            # 如果文件不存在，可以返回一个哨兵值或跳过该样本（取决于你的 collate_fn）
            # 这里我们返回一个空的 long 张量作为示例
            img_ids = torch.empty(0, dtype=torch.long)

        # --- 4. 根据公共索引加载文本 Token ---
        text_token_path = os.path.join(self.text_token_root, f"{common_idx_str}.npy")
        try:
            # 文本 Token IDs 也应为整数类型
            txt_ids = torch.from_numpy(np.load(text_token_path)).long()
        except FileNotFoundError:
            print(f"警告: 文本 Token 文件未找到: {text_token_path}。将返回一个空张量。")
            txt_ids = torch.empty(0, dtype=torch.long)

        # --- 5. 创建 micro_conds 张量 ---
        # 根据您的需求创建固定的 micro_conds 张量
        micro_conds = torch.tensor([512, 512, 0, 0, 6], dtype=torch.long)

        # --- 6. 组装并返回样本字典 ---
        sample_dict = {
            'fmri_data': fmri_data,  # (S_fmri, D_fmri)
            'img_ids': img_ids,  # (H, W) or (S_img, D_img_token)
            'txt_ids': txt_ids,  # (S_txt,) or (S_txt, D_txt_token)
            'micro_conds': micro_conds,  # 固定的 micro_conds 张量
        }

        return sample_dict


def collate_fn(samples):
    # 提取每个样本的各个字段
    fmri_data = [sample["fmri_data"] for sample in samples]
    img_ids = [sample["img_ids"] for sample in samples]
    txt_ids = [sample["txt_ids"] for sample in samples]
    micro_conds = [sample["micro_conds"] for sample in samples]

    # 堆叠可以堆叠的张量
    # 注意：fmri_data, img_ids, txt_ids可能有不同的形状，需要确保它们可以堆叠
    # 如果形状不一致，可能需要填充或使用其他方法
    try:
        fmri_data = torch.stack(fmri_data, dim=0)
    except RuntimeError:
        print("警告: fmri_data 形状不一致，无法堆叠")
        # 这里可以添加填充逻辑

    try:
        img_ids = torch.stack(img_ids, dim=0)
    except RuntimeError:
        print("警告: img_ids 形状不一致，无法堆叠")
        # 这里可以添加填充逻辑

    try:
        txt_ids = torch.stack(txt_ids, dim=0)
    except RuntimeError:
        print("警告: txt_ids 形状不一致，无法堆叠")
        # 这里可以添加填充逻辑

    micro_conds = torch.stack(micro_conds, dim=0)

    # 返回批处理后的数据
    return {
        'fmri_data': fmri_data,
        'img_ids': img_ids,
        'txt_ids': txt_ids,
        'micro_conds': micro_conds,
    }



def encode_prompt(
    text_encoder,
    input_ids,
    text_encoder_architecture='open_clip'
):
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        outputs = text_encoder(input_ids=input_ids, return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2]
        cond_embeds = outputs[0]
        return encoder_hidden_states, cond_embeds
    elif text_encoder_architecture == 't5_clip':
        outputs_clip = text_encoder[0](
            input_ids=input_ids[0],
            return_dict=True,
            output_hidden_states=True
        )
        outputs_t5 = text_encoder[1](
            input_ids=input_ids[1],
            return_dict=True,
            output_hidden_states=True
        )
        encoder_hidden_states = outputs_t5.last_hidden_state
        cond_embeds = outputs_clip.text_embeds
        return encoder_hidden_states, cond_embeds
    elif text_encoder_architecture == "gemma":
        outputs = text_encoder(**input_ids.to(text_encoder.device))
        encoder_hidden_states = outputs.last_hidden_states
        cond_embeds = encoder_hidden_states.mean(dim=-2)
        return encoder_hidden_states, cond_embeds
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")

def read_text_files_to_list(file_paths: List[str]) -> List[str]:
    """
    读取一个包含文本文件路径的列表，并返回一个包含这些文件内容的列表。

    Args:
        file_paths (List[str]): 一个包含多个.txt文件完整路径的列表。

    Returns:
        List[str]: 一个列表，其中每个元素都是对应文件中的文本内容（字符串）。
                   如果文件不存在或读取失败，将打印错误信息但不会中断程序。

    Raises:
        TypeError: 如果输入的 `file_paths` 不是一个列表。
    """
    # 检查输入是否为列表，保证函数的健壮性
    if not isinstance(file_paths, list):
        raise TypeError(f"输入参数必须是一个列表 (list)，但收到了 {type(file_paths)} 类型。")

    all_texts = []

    # 遍历列表中的每一个文件路径
    for file_path in file_paths:
        try:
            # 使用 'with open' 语句来安全地打开和关闭文件
            # 'r' 表示以只读模式打开
            # 'encoding="utf-8"' 是一个好习惯，可以处理各种字符，避免乱码
            with open(file_path, 'r', encoding='utf-8') as f:
                # f.read() 会读取整个文件的内容作为一个字符串
                # .strip() 会移除字符串开头和结尾可能存在的空白字符（如换行符、空格）
                content = f.read().strip()
                all_texts.append(content)

        except FileNotFoundError:
            # 如果文件路径不正确，打印错误信息
            print(f"警告：文件未找到，已跳过 -> {file_path}")
        except Exception as e:
            # 捕获其他可能的读取错误（如权限问题）
            print(f"警告：读取文件时发生错误，已跳过 -> {file_path}")
            print(f"错误详情: {e}")

    return all_texts

def load_npy_files_to_tensor(file_paths: List[str]) -> torch.Tensor:
    """
    读取一个包含.npy文件路径的列表，将它们加载为NumPy数组，
    然后将所有数组堆叠成一个单一的PyTorch Tensor。

    Args:
        file_paths (List[str]): 一个包含多个.npy文件完整路径的列表。
                                 函数假设每个.npy文件都包含一个一维数组，
                                 并且所有数组的长度（N）都相同。

    Returns:
        torch.Tensor: 一个形状为 (B, N) 的PyTorch Tensor。
                      B 是成功读取的文件数量。
                      N 是每个.npy文件内数组的长度。
                      如果没有任何文件被成功读取，将返回一个空的Tensor。

    Raises:
        TypeError: 如果输入的 `file_paths` 不是一个列表。
        ValueError: 如果成功加载的.npy数组形状不一致，无法堆叠。
    """
    # 检查输入是否为列表
    if not isinstance(file_paths, list):
        raise TypeError(f"输入参数必须是一个列表 (list)，但收到了 {type(file_paths)} 类型。")

    loaded_arrays = []

    # 遍历列表中的每一个文件路径
    for file_path in file_paths:
        try:
            # 使用 np.load() 读取 .npy 文件
            numpy_array = np.load(file_path)
            loaded_arrays.append(numpy_array)

        except FileNotFoundError:
            # 如果文件路径不正确，打印错误信息并跳过
            print(f"警告：文件未找到，已跳过 -> {file_path}")
        except Exception as e:
            # 捕获其他可能的读取错误（如文件损坏）
            print(f"警告：读取文件时发生错误，已跳过 -> {file_path}")
            print(f"错误详情: {e}")

    # 检查是否成功加载了任何数据
    if not loaded_arrays:
        print("警告：未能成功加载任何.npy文件，返回一个空的Tensor。")
        return torch.empty(0)  # 返回一个形状为 torch.Size([0]) 的空Tensor

    try:
        # 使用 torch.from_numpy 将NumPy数组列表高效地转换为PyTorch Tensor
        # np.stack(loaded_arrays) 会将数组列表堆叠成一个新的NumPy数组，形状为 (B, N)
        # torch.from_numpy() 会创建一个共享内存的Tensor，非常高效
        # .float() 确保最终的Tensor是浮点类型，这在深度学习中很常用
        return torch.from_numpy(np.stack(loaded_arrays)).float()

    except ValueError as e:
        # 如果数组形状不一致，np.stack会报错
        print(f"错误：加载的.npy文件形状不一致，无法堆叠成一个Tensor。请检查您的数据。")
        print(f"错误详情: {e}")
        # 抛出异常，让调用者知道发生了严重错误
        raise

def calculate_pcc_tensor(generated_fmri: torch.Tensor, val_brain: torch.Tensor) -> float:
    """
    计算两个形状为 (B, N) 的张量之间的平均皮尔逊相关系数 (PCC)。

    这个函数能自动处理输入张量位于不同设备（如CPU和GPU）上的情况。
    它会将两个张量都移动到 `generated_fmri` 所在的设备上进行计算。

    Args:
        generated_fmri (torch.Tensor): 生成或重建的fMRI数据张量。
                                       形状: (B, N)，其中B是批次大小, N是特征数。
        val_brain (torch.Tensor):      真实的fMRI数据（验证集）张量。
                                       形状: (B, N)。

    Returns:
        float: 一个浮点数，表示该批次中所有样本PCC分数的平均值。

    Raises:
        AssertionError: 如果输入张量的形状不匹配。
    """
    # --- 1. 验证和设备同步 ---
    # 确保输入张量的形状完全相同
    assert generated_fmri.shape == val_brain.shape, \
        f"输入张量的形状必须相同，但收到: {generated_fmri.shape} 和 {val_brain.shape}"

    # 确定计算设备（以 generated_fmri 的设备为准）
    target_device = generated_fmri.device

    # 将 val_brain 移动到与 generated_fmri 相同的设备上
    # non_blocking=True 是一个针对GPU的小优化，如果设备已经是目标设备，此操作几乎无开销
    val_brain = val_brain.to(target_device, non_blocking=True)

    # 确保数据类型为浮点数以进行精确计算
    x = generated_fmri.float()
    y = val_brain.float()

    # --- 2. 向量化计算PCC ---
    # 沿着特征维度(dim=1)计算每个样本的均值
    # keepdim=True 保持维度为 (B, 1) 以便进行广播减法
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)

    # 中心化数据 (x - x_mean)
    centered_x = x - mean_x
    centered_y = y - mean_y

    # 计算协方差项 (numerator)
    # 元素相乘后，沿着特征维度求和
    covariance = torch.sum(centered_x * centered_y, dim=1)

    # 计算标准差的乘积项 (denominator)
    bessel_correction_x = torch.sqrt(torch.sum(centered_x ** 2, dim=1))
    bessel_correction_y = torch.sqrt(torch.sum(centered_y ** 2, dim=1))
    denominator = bessel_correction_x * bessel_correction_y

    # --- 3. 计算PCC并返回平均值 ---
    # 计算每个样本的PCC
    # 添加一个很小的epsilon (1e-8) 来防止除以零，保证数值稳定性
    pcc_per_sample = covariance / (denominator + 1e-8)

    # 计算整个批次的平均PCC，并使用 .item() 将其从单元素Tensor转换为Python浮点数
    average_pcc = torch.mean(pcc_per_sample).item()

    return average_pcc



if __name__ == '__main__':
    """
    img_path = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_imgs/00000.png"
    image = Image.open(img_path).convert("RGB")
    image = exif_transpose(image)

    if not image.mode == "RGB":
        image = image.convert("RGB")
    size = 512

    orig_height = image.height
    orig_width = image.width

    image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)(image)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(size, size))
    print('')
    """

    # 设置数据路径 - 请根据您的实际情况修改这些路径
    fMRI_root = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_single/"
    image_token_root = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/VQVAE_feature_img/"
    text_token_root = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/caption_ids_COCO_recaption/"

    # 创建数据集
    dataset = Stage1_TrainDataset(
        fMRI_root=fMRI_root,
        image_token_root=image_token_root,
        text_token_root=text_token_root
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 使用较小的批次大小进行测试
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # 测试数据加载器
    print(">>> 开始测试数据加载器...")
    for i, batch in enumerate(dataloader):
        print(f"批次 {i}:")
        print(f"  fMRI数据形状: {batch['fmri_data'].shape}")
        print(f"  图像token形状: {batch['img_ids'].shape}")
        print(f"  文本token形状: {batch['txt_ids'].shape}")
        print(f"  micro_conds形状: {batch['micro_conds'].shape}")
        print(f"  micro_conds值: {batch['micro_conds']}")

        # 只打印前两个批次
        if i >= 1:
            break

    print(">>> 数据加载器测试完成!")
    """
    fMRI数据形状: torch.Size([4, 16127])
      图像token形状: torch.Size([4, 32, 32])
      文本token形状: torch.Size([4, 77])
      micro_conds形状: torch.Size([4, 5])
    """