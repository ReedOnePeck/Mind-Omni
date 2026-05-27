import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from dataclasses import dataclass
import json
from safetensors.torch import load_file
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
import torch
import PIL
import numpy as np

from train_stage1.train_stage1_utils import Stage1_TrainDataset, collate_fn, encode_prompt,  \
    load_npy_files_to_tensor
import collections
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from torch import nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import VQModel
from train_fMRI_tokenizer_perceptual.fMRI_tokenizer_perceptual import VQ_fMRI
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from MindOmni_utils.scheduler import Scheduler
from MindOmni_src_stage2.r_tri_modal_transformer import Trimodal_SymmetricTransformer2DModel

from MindOmni_src_stage2.r_tri_modal_pipeline import UnifiedPipeline
from PIL import Image
from train_stage2_short_VQA.lora_checkpoint_utils import ensure_lora_adapter_for_checkpoint

from torchvision.utils import save_image, make_grid


def load_images_to_tensor(image_paths, target_size=(512, 512), device='cpu'):
    """加载一系列图像路径，将它们转换为一个tensor batch。"""
    img_tensors = []
    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # 将PIL Image转为[C, H, W] tensor并归一化到[0, 1]
    ])
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_tensors.append(transform(img))
        except FileNotFoundError:
            print(f"警告: 图像文件未找到于 {path}, 将跳过。")
            continue
    if not img_tensors:
        return None
    # 将图像tensor列表堆叠成一个batch
    return torch.stack(img_tensors).to(device)


def read_text_files_to_list(text_paths):
    """读取一系列文本文件的内容到一个列表中。"""
    texts = []
    for path in text_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())
        except FileNotFoundError:
            print(f"警告: 文本文件未找到于 {path}, 将使用空字符串代替。")
            texts.append("")
    return texts


def calculate_pcc_tensor(x, y):
    """
    计算两个tensor之间的皮尔逊相关系数。
    假设 x 和 y 的形状都是 (n_samples, n_features)。
    函数会为每个样本计算PCC，然后返回所有样本PCC的平均值。
    """
    # 确保在同一设备上
    x = x.to(y.device)

    # 去均值
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    # 计算协方差
    covariance = (x_centered * y_centered).sum(dim=1)

    # 计算标准差的乘积
    x_std_dev = torch.sqrt((x_centered ** 2).sum(dim=1))
    y_std_dev = torch.sqrt((y_centered ** 2).sum(dim=1))

    # 计算PCC
    pcc = covariance / (x_std_dev * y_std_dev)

    # 返回所有样本PCC的平均值
    return pcc.mean().item()




import logging
from collections import OrderedDict
from typing import Dict, Any, Tuple

# --- 配置一个简单的日志记录器，用于在控制台输出信息 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def load_compiled_aware_checkpoint(
        model_to_load_into: torch.nn.Module,
        ckpt_path: str
) -> Tuple[list, list]:
    """
    智能加载 PyTorch 检查点，自动处理 torch.compile() 编译前后 state_dict 键名不匹配的问题。

    `torch.compile()` 会在模型参数的键名前添加 "_orig_mod." 前缀。此函数会自动检测
    目标模型和检查点文件是否经过编译，并相应地添加或移除该前缀，以确保成功加载权重。

    Args:
        model_to_load_into (torch.nn.Module): 已初始化的目标模型，权重将被加载到此模型中。
        ckpt_path (str): 权重检查点文件的路径。

    """
    ensure_lora_adapter_for_checkpoint(
        model=model_to_load_into,
        checkpoint_path=ckpt_path,
        fallback_lora_alpha=16,
        fallback_target_modules=["to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj",
                                 "to_add_out", "proj_mlp", "proj_out", "FFN_proj_in", "FFN_proj_out"],
        fallback_use_dora=True,
        logger=logger,
    )

    # 1. 加载检查点权重到CPU (推荐做法，避免GPU显存波动)
    try:
        loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
        logger.info(f"成功从路径 '{ckpt_path}' 加载检查点。")
    except FileNotFoundError:
        logger.error(f"错误：检查点文件未找到于 '{ckpt_path}'")
        return []
    except Exception as e:
        logger.error(f"加载检查点 '{ckpt_path}' 时发生错误: {e}")
        return []

    # ======================== [核心逻辑开始] ========================
    # 2. 智能处理编译和非编译模型之间的权重键名不匹配问题

    # 检查模型是否已编译 (其 state_dict 中的键是否以 "_orig_mod." 开头)
    model_is_compiled = any(k.startswith("_orig_mod.") for k in model_to_load_into.state_dict())
    # 检查加载的权重检查点(ckpt)是否来自一个编译过的模型
    ckpt_is_compiled = any(k.startswith("_orig_mod.") for k in loaded_state_dict.keys())

    final_state_dict_to_load = OrderedDict()

    # 根据模型和权重的状态进行智能调整
    if model_is_compiled and not ckpt_is_compiled:
        # 情况1: 模型已编译，但权重未编译 -> 给权重键添加前缀
        logger.info("检测到目标模型已编译，而检查点未编译。正在为权重键添加 '_orig_mod.' 前缀以进行匹配。")
        for key, value in loaded_state_dict.items():
            final_state_dict_to_load["_orig_mod." + key] = value

    elif not model_is_compiled and ckpt_is_compiled:
        # 情况2: 模型未编译，但权重已编译 -> 从权重键移除前缀
        logger.info("检测到目标模型未编译，而检查点已编译。正在从权重键移除 '_orig_mod.' 前缀以进行匹配。")
        for key, value in loaded_state_dict.items():
            if key.startswith("_orig_mod."):
                final_state_dict_to_load[key[len("_orig_mod."):]] = value
            else:
                final_state_dict_to_load[key] = value  # 保持其他没有前缀的键 (例如 buffer)

    else:
        # 情况3: 两者状态一致 (都编译了 或 都没编译)，无需操作
        if model_is_compiled:
            logger.info("检测到目标模型和检查点均已被编译。权重键格式匹配，无需调整。")
        else:
            logger.info("检测到目标模型和检查点均未被编译。权重键格式匹配，无需调整。")
        final_state_dict_to_load = loaded_state_dict

    # ======================== [核心逻辑结束] ========================

    # 3. 使用 strict=False 加载权重，并提供详细报告
    missing_keys, unexpected_keys = model_to_load_into.load_state_dict(final_state_dict_to_load, strict=False)

    # 4. 打印加载报告
    if missing_keys:
        logger.warning(f"加载权重时发现有 {len(missing_keys)} 个键缺失: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"加载权重时发现有 {len(unexpected_keys)} 个多余的键: {unexpected_keys}")

    if not missing_keys and not unexpected_keys:
        logger.info("权重成功加载，所有键完全匹配！")
    else:
        logger.info("权重加载完成，但存在不匹配的键。")

    return model_to_load_into


class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.indices = torch.arange(len(data))  # 生成0到len(data)-1的索引

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回数据和对应的原始索引
        return self.data[idx], self.indices[idx]


device = 'cuda:0'
tokenizer = CLIPTokenizer.from_pretrained("/data/home/luyizhuo/Datastation_lyz/Models/Muddit/tokenizer")
text_encoder = CLIPTextModelWithProjection.from_pretrained("/data/home/luyizhuo/Datastation_lyz/Models/Muddit/text_encoder")
text_encoder.requires_grad_(False)
text_encoder = text_encoder.to(device)


vq_model = VQModel.from_pretrained("/data/home/luyizhuo/Datastation_lyz/Models/Muddit/vqvae")
vq_model.requires_grad_(False)
vq_model = vq_model.to(device)

brain_vae = VQ_fMRI.from_pretrained(
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI")
brain_vae.requires_grad_(False)
brain_vae = brain_vae.to(device)



# --- 准备工作 ---


global_step = 1200  # 将步数设为一个变量，方便修改


checkpoint_path = f"/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage2_shortVQA/checkpoint-{global_step}"
config_path = os.path.join(checkpoint_path, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
model = Trimodal_SymmetricTransformer2DModel(**config)
scheduler = Scheduler.from_pretrained("/data/home/luyizhuo/Datastation_lyz/Models/Muddit/scheduler/")
model = load_compiled_aware_checkpoint(model_to_load_into=model,ckpt_path=f'/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage2_shortVQA/checkpoint-{global_step}/pytorch_model.bin')
model.requires_grad_(False)
model = model.to(device)
pipe = UnifiedPipeline(
                        transformer=model,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        vqvae=vq_model,
                        scheduler=scheduler,
                        brain_tokenizer=brain_vae
                            )

batch_size = 16


json_folder = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA'

output_dir = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage2_shortVQA'


Subs = [1]

for sub in Subs:
    test_fMRI_multi = np.load(f'/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub{sub}/sub{sub}_test_multi.npy')
    test_img_indices = np.load(f'/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub{sub}/test_img_index_start_from0.npy')

    valid_indices = []  # 存储test_img_indices中存在对应文件的位置索引（用于提取fMRI数据）
    valid_img_nums = []  # 存储存在对应文件的数字（test_img_indices中的值）

    print("筛选存在对应JSON文件的索引...")
    for idx, num in enumerate(test_img_indices):
        # 将数字格式化为5位字符串（如123 -> "00123"）
        json_filename = f"{num:05d}.json"
        json_path = os.path.join(json_folder, json_filename)

        # 检查文件是否存在
        if os.path.exists(json_path):
            valid_indices.append(idx)
            valid_img_nums.append(num)

    # 4. 根据有效索引提取fMRI数据
    filtered_fmri = test_fMRI_multi[np.array(valid_indices)]


    test_img_indices = np.array(valid_img_nums)

    # ====================================================================VQA====================================================
    text_save_path = os.path.join(output_dir, f'sub{sub}_step_{global_step}_shortcaption.jsonl')
    test_dataset = IndexedDataset(torch.tensor(filtered_fmri))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    all_generated_txts = []

    for batch in tqdm(test_loader, desc="[VQA] 简短描述测试集"):
        val_brain_batch, indices = batch  # 分别获取数据和索引
        val_brain_batch = val_brain_batch.to(device)
        idx = np.array(indices)  # 当前batch在test_fMRI_multi中的索引

        # 1. 根据idx从test_img_indices中获取对应的json文件编号
        json_indices = test_img_indices[idx]

        # 2. 处理问题列表
        batch_questions = []
        for json_idx in json_indices:
            # 转换为5位数字的文件名（如123 -> 00123.json）
            json_filename = f"{json_idx:05d}.json"
            json_path = os.path.join(json_folder, json_filename)

            # 读取json文件并提取问题
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    question = data[0]["Question"]  # 获取问题部分

                    # 清理问题文本
                    cleaned_question = question.replace("<image>", "").replace("\n", "")
                    batch_questions.append(cleaned_question)
            except Exception as e:
                print(f"处理文件 {json_filename} 时出错: {e}")
                # 出错时可以添加一个空字符串或其他占位符
                batch_questions.append("")



        # 使用 no_grad() 节省显存
        with torch.no_grad():
            output_batch = pipe(
                brain_data=val_brain_batch,
                prompt=batch_questions,
                is_BQA=True,
                num_brain_token=64,
                height=512,
                width=512,
                num_inference_steps=64,
                mask_token_embedding='/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth',
                brain_mask_token_path="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage2_mixed/fmri_mask_embedding.pt",
                generator=torch.manual_seed(42)
            )

        # 将结果移出循环，避免持有不必要的引用
        generated_txt_batch = output_batch.prompts
        all_generated_txts.extend(generated_txt_batch)
        del val_brain_batch, output_batch, generated_txt_batch

    # --- 3. 将所有生成的文本一次性保存到 .jsonl 文件 ---
    with open(text_save_path, "w", encoding='utf-8') as f:
        for prompt in all_generated_txts:
            # 将每个prompt作为一个独立的JSON对象写入，并添加换行符
            json_record = json.dumps({"prompt": prompt})
            f.write(json_record + '\n')

    print("\n推理完成！")
    print(f"所有生成的文本已保存于: {text_save_path}")

    # ====================================================================MM-Decoding====================================================
    # 创建保存结果的目录
    image_save_dir = os.path.join(output_dir, f'sub{sub}_step_{global_step}_images_MM')
    text_save_path = os.path.join(output_dir, f'sub{sub}_step_{global_step}_prompts_MM.jsonl')
    os.makedirs(image_save_dir, exist_ok=True)

    # --- 1. 加载并创建测试集 DataLoader ---
    # 将numpy数组转换为TensorDataset
    test_dataset = TensorDataset(torch.tensor(test_fMRI_multi))

    # 创建DataLoader以实现分批处理

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- 2. 循环推理并保存结果 ---
    all_generated_txts = []
    image_counter = 0

    # 使用tqdm来显示推理进度
    for batch in tqdm(test_loader, desc="[MM-Decoding] 推理测试集"):
        val_brain_batch = batch[0].to(device)

        # 使用 no_grad() 节省显存
        with torch.no_grad():
            output_batch = pipe(
                brain_data=val_brain_batch,
                num_brain_token=64,
                height=512,
                width=512,
                num_inference_steps=64,
                mask_token_embedding='/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth',
                brain_mask_token_path="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage2_mixed/fmri_mask_embedding.pt",
                generator=torch.manual_seed(42)
            )

        # 将结果移出循环，避免持有不必要的引用
        generated_txt_batch = output_batch.prompts
        generated_img_batch = output_batch.images
        all_generated_txts.extend(generated_txt_batch)
        for img in generated_img_batch:
            img_save_path = os.path.join(image_save_dir, f"{image_counter:04d}.png")
            img.save(img_save_path)
            image_counter += 1

        # [循环内清理] 删除此批次中创建的大张量
        del val_brain_batch, output_batch, generated_txt_batch, generated_img_batch

    # --- 3. 将所有生成的文本一次性保存到 .jsonl 文件 ---
    with open(text_save_path, "w", encoding='utf-8') as f:
        for prompt in all_generated_txts:
            # 将每个prompt作为一个独立的JSON对象写入，并添加换行符
            json_record = json.dumps({"prompt": prompt})
            f.write(json_record + '\n')

    print("\n推理完成！")
    print(f"共生成并保存了 {image_counter} 张图片于: {image_save_dir}")
    print(f"所有生成的文本已保存于: {text_save_path}")










