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
import collections
from torchvision import transforms


from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
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




device = 'cuda:0'
tokenizer = CLIPTokenizer.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/Muddit/tokenizer")
text_encoder = CLIPTextModelWithProjection.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/Muddit/text_encoder")
text_encoder.requires_grad_(False)
text_encoder = text_encoder.to(device)


vq_model = VQModel.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/Muddit/vqvae")
vq_model.requires_grad_(False)
vq_model = vq_model.to(device)

brain_vae = VQ_fMRI.from_pretrained(
    "/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI")
brain_vae.requires_grad_(False)
brain_vae = brain_vae.to(device)

checkpoint_path = "/data0/home/cddu/UniBrain/train_stage1/checkpoint-6000"
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

scheduler = Scheduler.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/Muddit/scheduler/")

pipe = UnifiedPipeline(
                        transformer=model,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        vqvae=vq_model,
                        scheduler=scheduler,
                        brain_tokenizer=brain_vae
                            )

sub = 5
test_img_index = np.load(f'/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub{sub}/test_img_index_start_from0.npy')
test_fMRI_multi = np.load(f'/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub{sub}/sub{sub}_test_multi.npy')


print(test_img_index[:9])
val_brain = torch.tensor(test_fMRI_multi[:9,:]).to(device)




output_dir = '/data0/home/cddu/UniBrain/train_stage1'


output1 = pipe(
                brain_data=val_brain,
                num_brain_token=64,
                height=512,
                width=512,
                num_inference_steps=64,
                mask_token_embedding='/nfs/diskstation/DataStation/ChangdeDu/Muddit/1024/mask_token_embedding.pth',
                brain_mask_token_path=os.path.join(output_dir, "fmri_mask_embedding.pt"),
                generator=torch.manual_seed(42)
                            )

generated_txt = output1.prompts
generated_img = output1.images

result = []
for img in generated_img:
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)
    result.append(img.unsqueeze(0))
result = torch.cat(result, dim=0)
result = make_grid(result, nrow=3)
img_save_path = os.path.join(output_dir, 'validation_results', f"validation_set_step_{6000}_brain2image.png")
save_image(result, img_save_path)

output_data = {
    "prompts": generated_txt,
}

with open(os.path.join(output_dir + '/validation_results', f"validation_set_step_{6000}_brain2text.json"), "w") as f:
    json.dump(output_data, f, indent=2)
