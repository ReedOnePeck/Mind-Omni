import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import random
import numpy as np
import h5py
from transformers import (
    CLIPTokenizer,
)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免图形界面问题

dtype = h5py.special_dtype(vlen=str)

MODEL_PATH = "/nfs/diskstation/DataStation/ChangdeDu/Muddit/CLIP-ViT-H-14-laion2B-s32B-b79K/"
CLIP_tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH)


def simple_token_count(batch_texts):
    """简化的 token 计数函数"""
    inputs = CLIP_tokenizer(
        batch_texts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    # 直接计算 attention mask 中 1 的数量
    token_counts = inputs.attention_mask.sum(dim=1).tolist()
    return token_counts


model = Qwen2VLForConditionalGeneration.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/qwen2-vl-7B",
                                                        torch_dtype=torch.bfloat16,
                                                        attn_implementation="flash_attention_2", ).to(
    "cuda")  # torch_dtype="auto"
processor = AutoProcessor.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/qwen2-vl-7B")
print("================================================================")

# 加载 captions 数据
BASE_DATA_PATH = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/"
CAPTIONS_PATH = os.path.join(BASE_DATA_PATH, 'COCO_73k_annots_curated.npy')

captions = np.load(CAPTIONS_PATH)
num_samples = len(captions)
print(f"Found {num_samples} samples.")

# 随机选择非空caption
print("Preprocessing: Randomly selecting one non-empty caption for each sample...")
selected_captions = []
for i in range(num_samples):
    valid_captions = [caption for caption in captions[i] if caption]
    if valid_captions:
        selected_captions.append(random.choice(valid_captions))
    else:
        selected_captions.append("")

# 创建输出目录
output_dir = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/COCO_captions_recapted_Qw2VL"
os.makedirs(output_dir, exist_ok=True)

prompt1 = 'Please rewrite the provided COCO caption into a slightly more detailed image description, but never longer than 40 words. Keep your sentences to a maximum of 40 words. When rewriting, please consider the following points: \
            1. Specify Objects: Provide detailed descriptions of the main objects in the image, including their color, material, condition, and actions.  \
            2. Maintain Natural Flow: Ensure the rewritten description is grammatically correct and expresses ideas fluently.  \
            3. Keep your sentences to a maximum of 40 words.  \
'

batch_texts = []
token_counts_list = []  # 存储所有token计数

# 使用tqdm创建进度条
for i in tqdm(range(73000), desc="Processing images"):
    try:
        img_path = f'/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_imgs/{str(i).zfill(5)}.png'
        prompt2 = f'Original COCO caption: {selected_captions[i]}'

        conversation1 = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt2},
                    {"type": "image", "image": img_path, "max_pixels": 512 * 512, },
                    {"type": "text", "text": prompt1},
                ],
            },
        ]

        with torch.no_grad():
            text = processor.apply_chat_template(
                conversation1, add_generation_prompt=True, add_vision_id=True
            )

            image_inputs, video_inputs = process_vision_info(conversation1)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference
            generated_ids = model.generate(**inputs, max_new_tokens=768)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 保存输出文本到文件
            output_file = os.path.join(output_dir, f"{str(i).zfill(5)}.txt")
            with open(output_file, 'w') as f:
                f.write(output_text)

            batch_texts.append(output_text)

            # 计算token计数并添加到列表
            token_count = simple_token_count([output_text])[0]
            token_counts_list.append(token_count)

    except Exception as e:
        print(f"Error processing image {i}: {e}")
        # 即使出错也继续处理下一个图像
        continue

# 保存token计数列表为npy文件
token_counts_array = np.array(token_counts_list)
token_counts_path = os.path.join(BASE_DATA_PATH, 'recaptioned_token_counts.npy')
np.save(token_counts_path, token_counts_array)
print(f"Token counts saved to {token_counts_path}")

# 绘制并保存直方图
plt.figure(figsize=(10, 6))
plt.hist(token_counts_list, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.title('Distribution of Token Counts in Recaptioned Captions')
plt.grid(True, alpha=0.3)

# 保存直方图
histogram_path = os.path.join(BASE_DATA_PATH, 'token_counts_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
print(f"Histogram saved to {histogram_path}")

# 打印一些统计信息
print(f"Total processed captions: {len(token_counts_list)}")
print(f"Average token count: {np.mean(token_counts_list):.2f}")
print(f"Median token count: {np.median(token_counts_list):.2f}")
print(f"Min token count: {np.min(token_counts_list)}")
print(f"Max token count: {np.max(token_counts_list)}")