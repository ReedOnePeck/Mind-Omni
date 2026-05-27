import numpy as np
import torch
from transformers import CLIPTokenizer
import os
from tqdm import tqdm
import random

# --- 1. 设置与初始化 ---
print(">>> 1. Setting up models, paths, and parameters...")

DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_PATH = "/nfs/diskstation/DataStation/ChangdeDu/Muddit/tokenizer/"
tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH)


# --- 2. 定义tokenize函数 ---
@torch.no_grad()
def tokenize_prompt(
        tokenizer,
        prompt,
        text_encoder_architecture='open_clip',
        padding='max_length',
        max_length=77,
):
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        input_ids = tokenizer(
            prompt,
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids
        return input_ids


# --- 3. 设置输入输出路径 ---
input_dir = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/COCO_captions_recapted_Qw2VL/"
output_dir = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/caption_ids_COCO_recaption/"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# --- 4. 处理所有txt文件 ---
print(">>> 2. Processing captions...")

# 创建文件列表
file_list = [f"{i:05d}.txt" for i in range(73000)]

# 使用tqdm创建进度条
for filename in tqdm(file_list, desc="Tokenizing captions"):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename.replace(".txt", ".npy"))

    # 读取caption
    with open(input_path, 'r') as f:
        caption = f.read().strip()

    # Tokenize
    input_ids = tokenize_prompt(tokenizer, caption, text_encoder_architecture='CLIP')

    # 转换为numpy数组并保存
    input_ids_np = input_ids.squeeze(0).numpy()  # 移除批次维度并转换为numpy
    np.save(output_path, input_ids_np)

print(">>> 3. All captions have been tokenized and saved!")

















