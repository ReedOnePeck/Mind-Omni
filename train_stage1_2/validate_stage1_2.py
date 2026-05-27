import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
from dataclasses import dataclass
import json
from safetensors.torch import load_file
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
import torch
import PIL
import numpy as np
import collections
from torchvision import transforms

from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)


from diffusers.image_processor import VaeImageProcessor
from diffusers.models import VQModel
from train_fMRI_tokenizer_perceptual.fMRI_tokenizer_perceptual import VQ_fMRI
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from MindOmni_utils.scheduler import Scheduler
from MindOmni_src.tri_modal_transformer import Trimodal_SymmetricTransformer2DModel
from MindOmni_src.tri_modal_pipeline import UnifiedPipeline
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm




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

checkpoint_path = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/checkpoint-24000"
config_path = f"{checkpoint_path}/config.json"
# 精确指定你的权重文件名！
weights_path = f"{checkpoint_path}/pytorch_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 手动从 config.json 加载配置并创建模型实例 ---
# 这一步绕过了 from_pretrained 的文件查找逻辑，直接根据配置创建模型结构
print("正在从 config.json 加载配置并创建模型...")
with open(config_path, "r") as f:
    config = json.load(f)

# 注意：你可能需要根据你的模型类的 __init__ 方法调整这里的参数传递
# 常见的方式是传递 **config 或 config 的某些值
# 假设你的模型可以直接接受配置字典中的所有键作为参数
# 如果 Trimodal_SymmetricTransformer2DModel 的 from_config 方法可用，优先使用
if hasattr(Trimodal_SymmetricTransformer2DModel, 'from_config'):
    model = Trimodal_SymmetricTransformer2DModel.from_config(config)
else:
    # 否则，尝试将config解包传入
    # 你需要确认你的模型 __init__ 函数需要哪些参数
    model = Trimodal_SymmetricTransformer2DModel(**config)

print("模型结构创建成功！")

# --- 3. 加载权重文件到CPU，并去除前缀 ---
print(f"正在从 {weights_path} 加载权重...")
state_dict_with_prefix = torch.load(weights_path, map_location="cpu")

new_state_dict = collections.OrderedDict()
prefix = "_orig_mod."
print(f"正在手动去除权重名称中的 '{prefix}' 前缀...")
for key, value in state_dict_with_prefix.items():
    if key.startswith(prefix):
        new_key = key[len(prefix):]
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value
print("前缀处理完毕！")

# --- 4. 将处理干净的权重加载到模型中 ---
print("正在将处理后的权重加载到模型...")
# 使用 load_state_dict 方法，这是 PyTorch 的标准操作
model.load_state_dict(new_state_dict)
model.requires_grad_(False)
model = model.to(device)

scheduler = Scheduler.from_pretrained("/data/home/luyizhuo/Datastation_lyz/Models/Muddit/scheduler/")

pipe = UnifiedPipeline(
                        transformer=model,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        vqvae=vq_model,
                        scheduler=scheduler,
                        brain_tokenizer=brain_vae
                            )









Subs = [2,5,7]
batch_size = 32
global_step = 24000



for sub in Subs:
    test_img_indices = np.load(f'/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub{sub}/test_img_index_start_from0.npy')
    test_fMRI_multi = np.load(f'/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub{sub}/sub{sub}_test_multi.npy')
    img_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs"
    txt_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL"

    output_dir = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage1_2/'

    if sub !=2 :
        #====================================================================MM-Decoding====================================================
        # 创建保存结果的目录
        image_save_dir = os.path.join(output_dir,  f'sub{sub}_step_{global_step}_images_MM')
        text_save_path = os.path.join(output_dir,  f'sub{sub}_step_{global_step}_prompts_MM.jsonl')
        os.makedirs(image_save_dir, exist_ok=True)

        test_dataset = TensorDataset(torch.tensor(test_fMRI_multi))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
                    brain_mask_token_path="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt",
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



    #====================================================================MM-Encoding====================================================
    val_brain = torch.tensor(test_fMRI_multi).to(device)
    num_samples = len(test_img_indices)

    all_generated_fmri_batches = []
    print(f"开始为被试 {sub} 进行分批次推理，共 {num_samples} 个样本，批大小为 {batch_size}...")

    # 按批次进行循环
    for i in tqdm(range(0, num_samples, batch_size), desc=f"[MM-Encoding] 为被试 {sub} 生成fMRI"):
        batch_indices = test_img_indices[i: i + batch_size]
        batch_image_paths = [os.path.join(img_dir, f"{idx:05d}.png") for idx in batch_indices]
        batch_text_paths = [os.path.join(txt_dir, f"{idx:05d}.txt") for idx in batch_indices]
        val_image_batch = load_images_to_tensor(batch_image_paths, target_size=(512, 512), device=device)
        val_text_batch = read_text_files_to_list(batch_text_paths)

        # 使用 no_grad() 节省显存
        with torch.no_grad():
            output = pipe(
                prompt=val_text_batch,
                image=val_image_batch,
                num_brain_token=64,
                height=512,
                width=512,
                num_inference_steps=64,
                mask_token_embedding='/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth',
                brain_mask_token_path="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt",
                generator=torch.manual_seed(42)
            )
            generated_fmri_token = output.brain
            generated_fmri_batch = brain_vae.decoder(
                brain_vae.post_quant_conv(generated_fmri_token.permute(0, 2, 1))
            )

        all_generated_fmri_batches.append(generated_fmri_batch.detach().cpu())

        # [循环内清理] 删除此批次中创建的所有不再需要的大张量
        del batch_indices, batch_image_paths, batch_text_paths, val_image_batch, val_text_batch
        del output, generated_fmri_token, generated_fmri_batch
    # =============================================================================
    # --- 4. 汇总结果，计算并保存PCC ---
    # =============================================================================

    # 将所有批次的生成结果拼接成一个大的tensor
    print("所有批次推理完成，正在汇总结果...")
    all_generated_fmri = torch.cat(all_generated_fmri_batches, dim=0)


    generated_fmri_numpy = all_generated_fmri.detach().cpu().numpy()
    fmri_save_path = os.path.join(output_dir, f'sub{sub}_step_{global_step}_generated_fmri_MM.npy')

    print(f"正在将生成的fMRI数据 ({generated_fmri_numpy.shape}) 保存为.npy文件...")
    np.save(fmri_save_path, generated_fmri_numpy)


    # 计算生成的fMRI和真实的fMRI之间的PCC
    print("正在计算平均皮尔逊相关系数 (PCC)...")
    final_recons_pcc = calculate_pcc_tensor(all_generated_fmri, val_brain.cpu())

    print(f"\n最终结果:")
    print(f"  - 被试: {sub}")
    print(f"  - 使用图文对做神经编码，平均重建PCC: {final_recons_pcc:.4f}")

    """
    最终结果:
      - 被试: 1
      - 使用图文对做神经编码，平均重建PCC: 0.1498
      - 被试: 1
      - 仅使用文本模态编码的平均重建PCC: 0.1013
      - 被试: 1
      - 仅使用图像模态编码的平均重建PCC: 0.1414
      
      
      被试: 5
    - 仅使用文本模态编码的平均重建PCC: 0.1259
    
    - 被试: 7
    - 使用图文对做神经编码，平均重建PCC: 0.1116
    - 被试: 7
    - 仅使用图像模态编码的平均重建PCC: 0.1060
    - 被试: 7
    - 仅使用文本模态编码的平均重建PCC: 0.0855
    """



    #====================================================================Image-Decoding====================================================
    image_save_dir = os.path.join(output_dir,  f'sub{sub}_step_{global_step}_images_only')
    os.makedirs(image_save_dir, exist_ok=True)

    test_dataset = TensorDataset(torch.tensor(test_fMRI_multi))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    image_counter = 0

    # 使用tqdm来显示推理进度
    for batch in tqdm(test_loader, desc="[Only-Decoding-Image] 推理测试集"):
        val_brain_batch = batch[0].to(device)

        # 使用 no_grad() 节省显存
        with torch.no_grad():
            output_batch = pipe(
                brain_data=val_brain_batch,
                is_brain_to_img_decoding = True,
                num_brain_token=64,
                height=512,
                width=512,
                num_inference_steps=64,
                mask_token_embedding='/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth',
                brain_mask_token_path="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt",
                generator=torch.manual_seed(42)
            )



        generated_img_batch = output_batch.images

        for img in generated_img_batch:
            img_save_path = os.path.join(image_save_dir, f"{image_counter:04d}.png")
            img.save(img_save_path)
            image_counter += 1

        # [循环内清理] 删除此批次中创建的大张量
        del val_brain_batch, output_batch,  generated_img_batch

    print("\n推理完成！")
    print(f"共生成并保存了 {image_counter} 张图片于: {image_save_dir}")


    #====================================================================Text-Decoding====================================================
    text_save_path = os.path.join(output_dir,  f'sub{sub}_step_{global_step}_prompts_only_text.jsonl')


    test_dataset = TensorDataset(torch.tensor(test_fMRI_multi))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_generated_txts = []

    # 使用tqdm来显示推理进度
    for batch in tqdm(test_loader, desc="[Only-Decoding-text] 推理测试集"):
        val_brain_batch = batch[0].to(device)

        # 使用 no_grad() 节省显存
        with torch.no_grad():
            output_batch = pipe(
                brain_data=val_brain_batch,
                is_brain_to_text_decoding=True,
                num_brain_token=64,
                height=512,
                width=512,
                num_inference_steps=64,
                mask_token_embedding='/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth',
                brain_mask_token_path="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt",
                generator=torch.manual_seed(42)
            )


        # 将结果移出循环，避免持有不必要的引用
        generated_txt_batch = output_batch.prompts
        all_generated_txts.extend(generated_txt_batch)


        # [循环内清理] 删除此批次中创建的大张量
        del val_brain_batch, output_batch, generated_txt_batch


    # --- 3. 将所有生成的文本一次性保存到 .jsonl 文件 ---
    with open(text_save_path, "w", encoding='utf-8') as f:
        for prompt in all_generated_txts:
            # 将每个prompt作为一个独立的JSON对象写入，并添加换行符
            json_record = json.dumps({"prompt": prompt})
            f.write(json_record + '\n')

    print("\n推理完成！")
    print(f"所有生成的文本已保存于: {text_save_path}")


    #====================================================================Image-Encoding====================================================

    val_brain = torch.tensor(test_fMRI_multi).to(device)
    num_samples = len(test_img_indices)

    all_generated_fmri_batches = []
    print(f"开始为被试 {sub} 进行分批次推理，共 {num_samples} 个样本，批大小为 {batch_size}...")

    # 按批次进行循环
    for i in tqdm(range(0, num_samples, batch_size), desc=f"[Only-img-Encoding] 为被试 {sub} 生成fMRI"):
        batch_indices = test_img_indices[i: i + batch_size]
        batch_image_paths = [os.path.join(img_dir, f"{idx:05d}.png") for idx in batch_indices]
        batch_text_paths = [os.path.join(txt_dir, f"{idx:05d}.txt") for idx in batch_indices]
        val_image_batch = load_images_to_tensor(batch_image_paths, target_size=(512, 512), device=device)
        val_text_batch = read_text_files_to_list(batch_text_paths)

        # 使用 no_grad() 节省显存
        with torch.no_grad():
            output = pipe(
                image=val_image_batch,
                is_img_to_brain_encoding=True,
                num_brain_token=64,
                height=512,
                width=512,
                num_inference_steps=64,
                mask_token_embedding='/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth',
                brain_mask_token_path="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt",
                generator=torch.manual_seed(42)
            )
            generated_fmri_token = output.brain
            generated_fmri_batch = brain_vae.decoder(
                brain_vae.post_quant_conv(generated_fmri_token.permute(0, 2, 1))
            )

        all_generated_fmri_batches.append(generated_fmri_batch.detach().cpu())

        # [循环内清理] 删除此批次中创建的所有不再需要的大张量
        del batch_indices, batch_image_paths, batch_text_paths, val_image_batch, val_text_batch
        del output, generated_fmri_token, generated_fmri_batch
    # =============================================================================
    # --- 4. 汇总结果，计算并保存PCC ---
    # =============================================================================

    # 将所有批次的生成结果拼接成一个大的tensor
    print("所有批次推理完成，正在汇总结果...")
    all_generated_fmri = torch.cat(all_generated_fmri_batches, dim=0)


    generated_fmri_numpy = all_generated_fmri.detach().cpu().numpy()
    fmri_save_path = os.path.join(output_dir, f'sub{sub}_step_{global_step}_generated_fmri_only_img.npy')

    print(f"正在将生成的fMRI数据 ({generated_fmri_numpy.shape}) 保存为.npy文件...")
    np.save(fmri_save_path, generated_fmri_numpy)


    # 计算生成的fMRI和真实的fMRI之间的PCC
    print("正在仅使用图像模态编码，计算平均皮尔逊相关系数 (PCC)...")
    final_recons_pcc = calculate_pcc_tensor(all_generated_fmri, val_brain.cpu())

    print(f"\n最终结果:")
    print(f"  - 被试: {sub}")
    print(f"  - 仅使用图像模态编码的平均重建PCC: {final_recons_pcc:.4f}")



    #====================================================================Text-Encoding====================================================

    val_brain = torch.tensor(test_fMRI_multi).to(device)
    num_samples = len(test_img_indices)

    all_generated_fmri_batches = []
    print(f"开始为被试 {sub} 进行分批次推理，共 {num_samples} 个样本，批大小为 {batch_size}...")

    # 按批次进行循环
    for i in tqdm(range(0, num_samples, batch_size), desc=f"[Ony-txt-Encoding] 为被试 {sub} 生成fMRI"):
        batch_indices = test_img_indices[i: i + batch_size]
        batch_image_paths = [os.path.join(img_dir, f"{idx:05d}.png") for idx in batch_indices]
        batch_text_paths = [os.path.join(txt_dir, f"{idx:05d}.txt") for idx in batch_indices]
        val_image_batch = load_images_to_tensor(batch_image_paths, target_size=(512, 512), device=device)
        val_text_batch = read_text_files_to_list(batch_text_paths)

        # 使用 no_grad() 节省显存
        with torch.no_grad():
            output = pipe(
                prompt=val_text_batch,
                is_text_to_brain_encoding=True,
                num_brain_token=64,
                height=512,
                width=512,
                num_inference_steps=64,
                mask_token_embedding='/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth',
                brain_mask_token_path="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt",
                generator=torch.manual_seed(42)
            )
            generated_fmri_token = output.brain
            generated_fmri_batch = brain_vae.decoder(
                brain_vae.post_quant_conv(generated_fmri_token.permute(0, 2, 1))
            )

        all_generated_fmri_batches.append(generated_fmri_batch.detach().cpu())

        # [循环内清理] 删除此批次中创建的所有不再需要的大张量
        del batch_indices, batch_image_paths, batch_text_paths, val_image_batch, val_text_batch
        del output, generated_fmri_token, generated_fmri_batch
    # =============================================================================
    # --- 4. 汇总结果，计算并保存PCC ---
    # =============================================================================

    # 将所有批次的生成结果拼接成一个大的tensor
    print("所有批次推理完成，正在汇总结果...")
    all_generated_fmri = torch.cat(all_generated_fmri_batches, dim=0)


    generated_fmri_numpy = all_generated_fmri.detach().cpu().numpy()
    fmri_save_path = os.path.join(output_dir, f'sub{sub}_step_{global_step}_generated_fmri_only_txt.npy')

    print(f"正在将生成的fMRI数据 ({generated_fmri_numpy.shape}) 保存为.npy文件...")
    np.save(fmri_save_path, generated_fmri_numpy)


    # 计算生成的fMRI和真实的fMRI之间的PCC
    print("正在仅使用文本模态编码，计算平均皮尔逊相关系数 (PCC)...")
    final_recons_pcc = calculate_pcc_tensor(all_generated_fmri, val_brain.cpu())

    print(f"\n最终结果:")
    print(f"  - 被试: {sub}")
    print(f"  - 仅使用文本模态编码的平均重建PCC: {final_recons_pcc:.4f}")


















