import argparse
import copy
import logging
import math
import os
from pathlib import Path
import sys
import random
sys.path.append(os.getcwd())
import json
import gc
from dataclasses import asdict, is_dataclass
import torch
import torch.nn.functional as F
from torch import nn
import json
from safetensors.torch import load_file
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import CLIPTextModelWithProjection, CLIPTokenizer
import diffusers.optimization
from diffusers import VQModel
from MindOmni_src.tri_modal_transformer import Trimodal_SymmetricTransformer2DModel
from train_fMRI_tokenizer_perceptual.fMRI_tokenizer_perceptual import VQ_fMRI
from collections import OrderedDict

from MindOmni_utils.scheduler import Scheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import is_wandb_available
from MindOmni_src.tri_modal_pipeline import UnifiedPipeline
from torchvision.utils import save_image, make_grid
from MindOmni_utils.trainer_utils import save_checkpoint

from train_stage1.train_stage1_utils import Stage1_TrainDataset, collate_fn, encode_prompt, read_text_files_to_list, \
    load_npy_files_to_tensor, calculate_pcc_tensor

from MindOmni_utils.trainer_utils import load_images_to_tensor

from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")

import torch._dynamo

torch._dynamo.config.verbose = True

# Optionally suppress errors to fall back to eager execution
torch._dynamo.config.suppress_errors = False

from accelerate.utils import DistributedDataParallelKwargs



def get_current_mode():
    """根据2:2:1的概率比返回当前模式"""
    # 定义模式列表
    modes = ["brain-image", "brain-text", "brain-image-text"]
    # 生成0-1之间的随机数
    rand_val = random.random()

    # 根据概率区间判断模式（总权重2+2+1=5）
    if rand_val < 2 / 5:  # 前2/5概率：brain-image
        return modes[0]
    elif rand_val < (2 + 2) / 5:  # 中间2/5概率：brain-text
        return modes[1]
    else:  # 最后1/5概率：brain-image-text
        return modes[2]


def load_pretrained_weights_for_trimodal_model(
        config_path: str,
        pretrained_weights_path: str,
        third_modal_codebook_size: int,
        third_modal_token_dim: int
) -> "Trimodal_SymmetricTransformer2DModel":
    """
    加载双模态预训练权重来初始化三模态模型。
    新增的第三模态模块会尝试从其对应的文本模态模块复制权重。

    Args:
        config_path (str): 原版模型 config.json 的路径。
        pretrained_weights_path (str): 原版模型 .safetensors 权重的路径。
        third_modal_codebook_size (int): 第三模态的 codebook 大小。
        third_modal_token_dim (int): 第三模态的 token 维度。

    Returns:
        Trimodal_SymmetricTransformer2DModel: 初始化完成的三模态模型。
    """
    print("--- 开始初始化三模态模型 ---")

    # 1. 加载原版 config 并添加新参数
    print(f"从 '{config_path}' 加载原始 config...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 从 config 中移除 _class_name 和 _diffusers_version，因为我们的类名不同
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)

    config['third_modal_codebook_size'] = third_modal_codebook_size
    config['third_modal_token_dim'] = third_modal_token_dim
    print("已向 config 添加第三模态参数。")

    # 2. 创建新的三模态模型 (此时为随机初始化)
    trimodal_model = Trimodal_SymmetricTransformer2DModel(**config)
    new_model_state_dict = trimodal_model.state_dict()
    print("新的三模态模型已创建 (随机初始化)。")

    # 3. 加载预训练的双模态权重
    print(f"从 '{pretrained_weights_path}' 加载预训练权重...")
    pretrained_state_dict = load_file(pretrained_weights_path)
    print(f"成功加载 {len(pretrained_state_dict)} 个预训练权重张量。")

    # 4. 将预训练权重复制到新模型中
    loaded_keys = []
    skipped_keys = []
    for key, value in pretrained_state_dict.items():
        if key in new_model_state_dict and new_model_state_dict[key].shape == value.shape:
            new_model_state_dict[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    # 5. 识别新增模块，并尝试从文本模态复制权重
    print("\n--- 开始处理新增模块的权重初始化 ---")

    # 定义命名映射规则: (新增模块名中的关键词, 文本模块名中的关键词)
    name_mappings = [
        ('third_modal_embedder', 'context_embedder'),
        ('third_modal_norm', 'encoder_proj_layer_norm'),
        ('third_modal_decoder', 'text_decoder'),
        ('norm1_thirdmodal', 'norm1_context'),
        ('norm2_thirdmodal', 'norm2_context'),
        ('ff_thirdmodal', 'ff_context'),
        ('third_add_q_proj', 'add_q_proj'),
        ('third_add_k_proj', 'add_k_proj'),
        ('third_add_v_proj', 'add_v_proj'),
        ('norm_thirdmodal_q', 'norm_added_q'),
        ('norm_thirdmodal_k', 'norm_added_k'),
        ('to_thirdmodal_out', 'to_add_out'),
    ]

    all_new_keys = [key for key in new_model_state_dict if key not in pretrained_state_dict]
    copied_from_text_keys = []
    randomly_initialized_keys = []

    for new_key in all_new_keys:
        was_copied = False
        for third_name, text_name in name_mappings:
            if third_name in new_key:
                # 尝试生成对应的文本模块权重键名
                text_counterpart_key = new_key.replace(third_name, text_name)

                # 检查文本模块的权重是否存在且形状匹配
                if text_counterpart_key in new_model_state_dict:
                    source_tensor = new_model_state_dict[text_counterpart_key]
                    target_tensor = new_model_state_dict[new_key]
                    if source_tensor.shape == target_tensor.shape:
                        # 执行复制
                        new_model_state_dict[new_key] = source_tensor.clone()
                        copied_from_text_keys.append((new_key, text_counterpart_key))
                        was_copied = True
                        break  # 已找到匹配，跳出内层循环

        if not was_copied:
            randomly_initialized_keys.append(new_key)

    # (可选) 打印识别出的文本模块名供人工检查
    print("\n--- 供人工检查：识别出的文本模态相关权重 ---")
    identified_text_keys = set()
    text_keywords = ['context', 'encoder', 'add_q', 'add_k', 'add_v', 'norm_added', 'to_add_out']
    for key in pretrained_state_dict:
        if any(keyword in key for keyword in text_keywords):
            identified_text_keys.add(key.split('.')[0] if '.' in key else key)  # 只打印顶层模块名

    # for module_name in sorted(list(identified_text_keys)):
    # print(f"  - {module_name}")

    # 6. 将最终更新后的 state_dict 加载回模型
    trimodal_model.load_state_dict(new_model_state_dict)

    # 7. 打印详细的加载总结
    print("\n--- 权重加载总结 ---")
    print(f"✅ 从预训练文件加载了 {len(loaded_keys)} 个权重。")
    print(f"🔄 从文本模态复制初始化了 {len(copied_from_text_keys)} 个新增权重。")
    print(f"✨ 保留了 {len(randomly_initialized_keys)} 个新增权重的随机初始化。")

    if skipped_keys:
        print(f"⚠️ 跳过了 {len(skipped_keys)} 个预训练权重 (名称或形状不匹配):")
        # for key in skipped_keys: print(f"  - {key}")
    else:
        print('✅ 所有预训练权重都已成功匹配并加载。')

    print("\n--- 从文本模态复制的权重详情 (新模块 <- 源模块) ---")
    if not copied_from_text_keys:
        print("  - 无")
    # for new_key, src_key in copied_from_text_keys:
    # print(f"  - {new_key}  <--  {src_key}")

    print("\n--- 保持随机初始化的权重详情 ---")
    if not randomly_initialized_keys:
        print("  - 无 (所有新增模块都成功从对应模块复制了权重)")
    # for key in randomly_initialized_keys:
    # print(f"  - {key}")

    print("\n--- 初始化完成！---")
    return trimodal_model


def freeze_pretrained_and_enable_new_modal_grads(model: Trimodal_SymmetricTransformer2DModel) -> list:
    """
    冻结模型中所有预训练的参数，只解冻与新增的第三模态相关的参数。

    Args:
        model (Trimodal_SymmetricTransformer2DModel): 已经加载了权重的三模态模型。

    Returns:
        list: 一个包含所有被解冻、可训练的参数对象的列表，可直接用于优化器。
    """
    print("\n--- 开始设置模型参数梯度 ---")

    # 1. 首先，冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    print("所有模型参数已被冻结。")

    # 2. 定义与第三模态相关的模块的关键词
    # 这些关键词应该与你在权重初始化时识别新增模块的逻辑一致
    third_modal_keywords = [
        'third_modal_embedder',
        'third_modal_norm',
        'third_modal_decoder',
        'norm1_thirdmodal',
        'norm2_thirdmodal',
        'ff_thirdmodal',
        'third_add_q_proj',
        'third_add_k_proj',
        'third_add_v_proj',
        'norm_thirdmodal_q',
        'norm_thirdmodal_k',
        'to_thirdmodal_out',
    ]

    # 3. 遍历所有命名参数，解冻包含关键词的参数
    trainable_params = []
    trainable_param_names = []

    for name, param in model.named_parameters():
        is_trainable = False
        for keyword in third_modal_keywords:
            if keyword in name:
                param.requires_grad = True
                is_trainable = True
                break  # 找到一个关键词匹配即可，无需继续检查

        if is_trainable:
            trainable_params.append(param)
            trainable_param_names.append(name)

    # 4. 打印总结报告
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)

    print(f"已解冻 {len(trainable_param_names)} 个与第三模态相关的参数张量。")
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params_count / 1e6:.2f} M")
    print(f"可训练参数占比: {100 * trainable_params_count / total_params:.2f}%")

    # print("\n--- 可训练参数列表 ---")
    # for name in trainable_param_names:
    # print(f"  - {name}")

    print("\n--- 参数梯度设置完成！---")

    return trainable_params


def load_transformer_checkpoint_weights(model_to_load_into, weights_path, logger):
    loaded_state_dict = torch.load(weights_path, map_location="cpu")
    logger.info(f"正在从 {weights_path} 加载完整的模型权重...")

    model_is_compiled = any(k.startswith("_orig_mod.") for k in model_to_load_into.state_dict())
    ckpt_is_compiled = any(k.startswith("_orig_mod.") for k in loaded_state_dict.keys())
    final_state_dict_to_load = OrderedDict()

    if model_is_compiled and not ckpt_is_compiled:
        logger.info("检测到目标模型已编译，而检查点未编译。正在为权重键添加 '_orig_mod.' 前缀以进行匹配。")
        for key, value in loaded_state_dict.items():
            final_state_dict_to_load["_orig_mod." + key] = value
    elif not model_is_compiled and ckpt_is_compiled:
        logger.info("检测到目标模型未编译，而检查点已编译。正在从权重键移除 '_orig_mod.' 前缀以进行匹配。")
        for key, value in loaded_state_dict.items():
            if key.startswith("_orig_mod."):
                final_state_dict_to_load[key[len("_orig_mod."):]] = value
            else:
                final_state_dict_to_load[key] = value
    else:
        if model_is_compiled:
            logger.info("检测到目标模型和检查点均已被编译。权重键格式匹配，无需调整。")
        else:
            logger.info("检测到目标模型和检查点均未被编译。权重键格式匹配，无需调整。")
        final_state_dict_to_load = loaded_state_dict

    missing_keys, unexpected_keys = model_to_load_into.load_state_dict(final_state_dict_to_load, strict=False)

    logger.info("权重加载完成。")
    if unexpected_keys:
        logger.warning(f"在检查点中发现 {len(unexpected_keys)} 个意外的键未被加载: {unexpected_keys[:5]}...")
    if missing_keys:
        logger.warning(f"模型中有 {len(missing_keys)} 个键未在检查点中找到: {missing_keys[:5]}...")
    if not unexpected_keys and not missing_keys:
        logger.info("成功匹配并加载了所有模型权重！")


def parse_args():
    parser = argparse.ArgumentParser()
    # ---------------------------Model----------------------------------------
    parser.add_argument(
        "--muddit_model_config",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/512/transformer/config.json",
        required=False,
    )
    parser.add_argument(
        "--muddit_model_ckpt",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/512/transformer/diffusion_pytorch_model.safetensors",
        required=False,
    )
    parser.add_argument(
        "--muddit_model_scheduler",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/scheduler/",
        required=False,
    )
    parser.add_argument(
        "--img_vae_model_ckpt",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/vqvae",
        required=False,
    )
    parser.add_argument(
        "--text_encoder_model_ckpt",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/text_encoder",
        required=False,
    )
    parser.add_argument(
        "--text_tokenizer_ckpt",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/tokenizer",
        required=False,
    )
    parser.add_argument(
        "--txt_mask_token_file",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth",
        required=False,
    )
    parser.add_argument(
        "--brain_vae_model_ckpt",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI",
        required=False,
    )
    parser.add_argument(
        "--brain_vae_codebook_size",
        type=int,
        default=128,
        required=False,
    )
    parser.add_argument(
        "--brain_vae_token_dim",
        type=int,
        default=16,
        required=False,
    )
    parser.add_argument(
        "--num_of_brain_token",
        type=int,
        default=64,
        required=False,
    )

    # ---------------------------Data----------------------------------------
    parser.add_argument(
        "--fMRI_single_trial",
        type=str,
        default='/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_single/',
        required=False,
    )
    parser.add_argument(
        "--img_token_ids",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/VQVAE_feature_img/",
        required=False,
    )
    parser.add_argument(
        "--txt_token_ids",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/caption_ids_COCO_recaption/",
        required=False,
    )
    parser.add_argument(
        "--fmri_mask_token_path",
        type=str,
        default="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_with_encoding/fmri_mask_embedding.pt",
        required=False,
        help="Path to the saved fMRI mask token embedding used during training and validation.",
    )

    # ---------------------------Train model----------------------------------------
    parser.add_argument(
        "--encoding_loss_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--training_from_scratch",
        type=bool,
        default=True,
        required=False
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/train_stage1/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=20020816, help="A seed for reproducible training.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5000,
        help=(
            "Run validation every X steps."
        ),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument("--min_masking_rate", type=float, default=0.0)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.", required=False)
    parser.add_argument("--use_lora", action="store_true", help="Fine tune the model using LoRa")
    parser.add_argument("--text_encoder_use_lora", action="store_true", help="Fine tune the model using LoRa")
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")
    parser.add_argument("--text_encoder_lora_r", default=16, type=int)
    parser.add_argument("--text_encoder_lora_alpha", default=32, type=int)
    parser.add_argument("--text_encoder_lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--image_to_text_only", action="store_true")
    parser.add_argument("--image_key", type=str, required=False)
    parser.add_argument("--prompt_key", type=str, required=False)

    # ---------------------------Validate model----------------------------------------
    parser.add_argument("--val_imgs", nargs='+', type=str, required=False, help="List of validation image paths.")
    parser.add_argument("--val_text", nargs='+', type=str, required=False, help="List of validation text file paths.")
    parser.add_argument("--val_brain", nargs='+', type=str, required=False, help="List of validation brain data paths.")

    args = parser.parse_args()

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    return args


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def prepare_brain_ids(brain_sequence_length: int, device, dtype) -> torch.Tensor:
    """
    为大脑信号 token 创建位置 ID。
    将 brain token 序列视为沿着 't' 轴排列。
    坐标格式: (t, h, w)
    """
    # 1. 创建一个 (brain_seq_len, 3) 的零张量
    brain_ids = torch.zeros(brain_sequence_length, 3, dtype=dtype, device=device)

    # 2. 填充 't' 坐标 (第一个维度)
    #    每个 token 的 t 坐标就是它在序列中的索引
    brain_ids[:, 0] = torch.arange(brain_sequence_length, device=device, dtype=dtype)

    # h 和 w 坐标保持为 0
    return brain_ids


def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    print("ddp_kwargs:", ddp_kwargs)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]  # <-- 在这里传入
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #    datefmt="%m/%d/%Y %H:%M:%S",
    #    level=logging.INFO,
    # )

    # --- 新的、更健壮的日志配置，支持同时输出到终端和文件 ---

    # 1. 获取 root logger
    #    我们直接对根记录器进行配置，这样所有地方的 logging 调用都会遵循这个规则。
    logger = logging.getLogger(__name__)  # 获取当前模块的 logger
    logger.setLevel(logging.INFO)  # 设置 logger 的最低响应级别

    # 2. 创建格式化器 (Formatter)
    #    定义所有日志消息的统一格式。
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    # 3. 创建并配置终端处理器 (StreamHandler)
    #    这个 handler 负责将日志打印到你的控制台。
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 4. 只在主进程上创建并配置件处理器 (FileHandler)
    #    这个 handler 负责将日志写入到文件，我们不希望每个 GPU 进程都写一遍。
    if accelerator.is_main_process:
        # 定义你想要的日志文件路径
        log_file_path = os.path.join(args.output_dir, "stage1_training_log.txt")
        print(f"Text logs will be saved to: {log_file_path}")

        # 'a' 模式表示追加 (append)，如果文件已存在，新日志会添加到末尾。
        # 'w' 模式表示写入 (write)，每次运行都会覆盖旧文件。
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # --- 现在，后续所有的 logger.info() 调用都会同时作用于终端和文件 ---

    logger.info(accelerator.state)

    if accelerator.is_main_process:
        accelerator.init_trackers("stage1_training", config=vars(copy.deepcopy(args)))

    if args.seed is not None:
        set_seed(args.seed)

    val_image = load_images_to_tensor(args.val_imgs, target_size=(512, 512))
    val_text = read_text_files_to_list(args.val_text)
    val_brain = load_npy_files_to_tensor(args.val_brain)

    tokenizer = CLIPTokenizer.from_pretrained(args.text_tokenizer_ckpt)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.text_encoder_model_ckpt)
    text_encoder.requires_grad_(False)

    extra_id_0_token = "<extra_id_0>"
    if extra_id_0_token in tokenizer.get_vocab():
        print(f"Token '{extra_id_0_token}' 已存在于 tokenizer 中。")
        clip_mask_id = tokenizer.convert_tokens_to_ids(extra_id_0_token)
        print(f"  - Token ID: {clip_mask_id}")
        print("  - 无需修改 tokenizer 和 text_encoder。")
    else:
        print(f"Token '{extra_id_0_token}' 不在 tokenizer 中，正在添加...")
        num_added_tokens = tokenizer.add_tokens(extra_id_0_token)
        if num_added_tokens == 0:
            raise RuntimeError(f"尝试添加 '{extra_id_0_token}' 失败，tokenizer.add_tokens 返回 0。")

        clip_mask_id = tokenizer.convert_tokens_to_ids(extra_id_0_token)
        text_encoder.resize_token_embeddings(len(tokenizer))
        mask_token_embedding = torch.load(args.txt_mask_token_file, map_location="cpu")

        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[clip_mask_id] = mask_token_embedding.to(
                device=text_encoder.device,
                dtype=text_encoder.get_input_embeddings().weight.dtype
            )

    vq_model = VQModel.from_pretrained(args.img_vae_model_ckpt)
    vq_model.requires_grad_(False)

    brain_vae = VQ_fMRI.from_pretrained(args.brain_vae_model_ckpt)
    brain_vae.requires_grad_(False)

    model = load_pretrained_weights_for_trimodal_model(
        config_path=args.muddit_model_config,
        pretrained_weights_path=args.muddit_model_ckpt,
        third_modal_codebook_size=args.brain_vae_codebook_size,
        third_modal_token_dim=args.brain_vae_token_dim
    )

    img_mask_id = model.config.vocab_size - 1
    img_codebook_size = model.config.codebook_size
    tokenizer_vocab_size = model.config.tokenizer_vocab_size

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
        )
        model.add_adapter(lora_config)

    model.train()
    model_params_to_train = freeze_pretrained_and_enable_new_modal_grads(model)

    model = torch.compile(model)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if len(models) > 0:
                # 获取被 accelerator 包装的原始模型对象
                model_to_save = accelerator.unwrap_model(models[0])

                # 定义保存模型的子目录
                model_output_dir = output_dir
                os.makedirs(model_output_dir, exist_ok=True)

                # 1. 保存模型的 state_dict (权重)
                torch.save(model_to_save.state_dict(), os.path.join(model_output_dir, "pytorch_model.bin"))

                # 2. 保存模型的 config (配置)
                #    我们假设模型的 config 是一个 dataclass
                if hasattr(model_to_save, 'config') and model_to_save.config is not None:
                    # 将 dataclass 转换为可以序列化为 JSON 的字典
                    config_dict = model_to_save.config
                    with open(os.path.join(model_output_dir, "config.json"), 'w') as f:
                        json.dump(config_dict, f, indent=2)

                print(f"Custom model state_dict and config saved to {model_output_dir}")

            # 弹出权重，告知 accelerator 已处理
            while len(weights) > 0:
                weights.pop()



    def load_model_hook(models, input_dir):
        if not models:
            return
        model_to_load_into = models.pop()
        weights_path = os.path.join(input_dir, "pytorch_model.bin")

        if not os.path.exists(weights_path):
            logger.warning(f"检查点权重文件未在 {weights_path} 找到。将跳过加载模型权重。")
            models.append(model_to_load_into)
            return

        load_transformer_checkpoint_weights(model_to_load_into, weights_path, logger)

        # 4. 将加载好权重的模型放回列表
        models.append(model_to_load_into)


    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer_grouped_parameters = [
        {
            "params": model_params_to_train,
            "weight_decay": args.adam_weight_decay,
        }
    ]

    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    dataset = Stage1_TrainDataset(
        fMRI_root=args.fMRI_single_trial,
        image_token_root=args.img_token_ids,
        text_token_root=args.txt_token_ids
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    train_dataloader.num_batches = len(train_dataloader)

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )

    logger.info("Preparing model, optimizer and dataloaders")

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    train_dataloader.num_batches = len(train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(device=accelerator.device)  # , dtype=weight_dtype)
    vq_model.to(device=accelerator.device)  # , dtype=weight_dtype)
    brain_vae.to(device=accelerator.device)  # , dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                resume_from_checkpoint = None

        if resume_from_checkpoint is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {resume_from_checkpoint}")

    if resume_from_checkpoint is None:
        global_step = 0
        first_epoch = 0
    else:
        checkpoint_is_in_output_dir = os.path.abspath(resume_from_checkpoint).startswith(
            os.path.abspath(args.output_dir)
        )
        if checkpoint_is_in_output_dir:
            accelerator.load_state(resume_from_checkpoint)
            global_step = int(os.path.basename(resume_from_checkpoint).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            weights_path = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
            load_transformer_checkpoint_weights(accelerator.unwrap_model(model), weights_path, logger)
            logger.info("检测到跨阶段 warm-start 检查点；仅加载模型权重，不恢复优化器/随机状态。")
            global_step = 0
            first_epoch = 0

    for epoch in range(first_epoch, num_train_epochs):
        progress_bar = tqdm(
            range(num_update_steps_per_epoch),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{num_train_epochs}"
        )

        for batch in train_dataloader:
            torch.cuda.empty_cache()
            with torch.no_grad():
                micro_conds = batch["micro_conds"].to(accelerator.device, non_blocking=True)
                encoding_micro_conds, decoding_micro_conds = micro_conds.chunk(2, dim=0)

                fMRI_data = batch["fmri_data"].to(accelerator.device, non_blocking=True)
                quantized_fmri_tokens, codebook_indices = brain_vae.forward_for_inference(fMRI_data)
                encoding_fMRI_data, decoding_fMRI_data = quantized_fmri_tokens.chunk(2, dim=0)
                encoding_fMRI_ids, _ = codebook_indices.chunk(2, dim=0)

                image_tokens = batch["img_ids"].reshape(fMRI_data.shape[0], -1).to(accelerator.device,
                                                                                   non_blocking=True)

                # 注意，text的前半部分没有被mask，只有后半部分是被mask的，因此前半部分对应encoding，对于图像模态也应该统一成前半部分完整，后半部分mask
                prompt_input_ids = batch["txt_ids"].to(accelerator.device, non_blocking=True)
                prompt_input_ids_clip_1, prompt_input_ids_clip_2 = prompt_input_ids.chunk(2, dim=0)
                encoder_hidden_states, cond_embeds = encode_prompt(
                    text_encoder,
                    prompt_input_ids_clip_1
                )
                encoder_hidden_states = encoder_hidden_states.to(accelerator.device, dtype=weight_dtype)
                cond_embeds = cond_embeds.to(accelerator.device, dtype=weight_dtype)

            # ====================== image perturbation   mask后半部分 ======================
            image_tokens_1, image_tokens_2 = image_tokens.chunk(2, dim=0)  # (b // 2, seq_len)
            half_batch_size, seq_len = image_tokens_2.shape
            sigma = torch.rand(half_batch_size, device=image_tokens_2.device)
            image_mask_prob = torch.cos(sigma * math.pi * 0.5)
            image_mask_prob = image_mask_prob.clip(args.min_masking_rate)
            image_timestep = image_mask_prob.clone().clamp(min=1e-3)

            num_token_masked = (seq_len * image_mask_prob).round().clamp(min=1)
            batch_randperm = torch.rand(half_batch_size, seq_len, device=image_tokens_2.device).argsort(dim=-1)
            mask = batch_randperm < num_token_masked.unsqueeze(-1)
            # ('vocab_size', 8256), ('codebook_size', 8192)
            mask_id = img_mask_id
            masked_image_ids = torch.where(mask, mask_id,
                                           image_tokens_2)  # 如果 `mask` 中某个位置是 `True`，就从 `mask_id` 中取值；如果是 `False`，就从原始的 `image_tokens_1` 中取值。
            image_labels = torch.where(mask, image_tokens_2, -100)

            # reshape to (batch size, channel, height, width)
            vae_scale_factor = 2 ** (len(vq_model.config.block_out_channels) - 1)
            resolution = 512 // vae_scale_factor
            masked_image_ids = masked_image_ids.reshape(half_batch_size, resolution, resolution)
            image_ids = image_tokens_1.reshape(half_batch_size, resolution, resolution)

            # ====================create pure_mask_image_latents for "brain-text"==========================================
            pure_mask_img_shape = (half_batch_size, resolution, resolution)
            pure_mask_image_latents = torch.full(pure_mask_img_shape, img_mask_id, dtype=torch.long,
                                                 device=image_tokens_2.device)
            # ========================================================================================================================
            # ====================== image perturbation ======================

            # ====================== text perturbation   mask后半部分======================
            half_batch_size, seq_len = prompt_input_ids_clip_2.shape
            # text和img共用一个sigma，即以相同的概率mask
            text_mask_prob = torch.cos(sigma * math.pi * 0.5)
            text_mask_prob = text_mask_prob.clip(args.min_masking_rate)
            text_timestep = text_mask_prob.clone().clamp(min=1e-3)

            num_token_masked = (seq_len * text_mask_prob).round().clamp(min=1)
            batch_randperm = torch.rand(half_batch_size, seq_len, device=image_tokens_1.device).argsort(dim=-1)
            mask = batch_randperm < num_token_masked.unsqueeze(-1)

            masked_prompt_input_ids_clip = torch.where(mask, clip_mask_id, prompt_input_ids_clip_2)
            text_labels = torch.where(mask, prompt_input_ids_clip_2, -100)
            # ====================== text perturbation ======================

            # ====================== encode masked text prompts ======================
            with torch.no_grad():
                masked_encoder_hidden_states, masked_cond_embeds = encode_prompt(
                    text_encoder,
                    masked_prompt_input_ids_clip
                )
                masked_encoder_hidden_states = masked_encoder_hidden_states.to(accelerator.device, dtype=weight_dtype)
                masked_cond_embeds = masked_cond_embeds.to(accelerator.device, dtype=weight_dtype)

                # ====================== encode pure masked text prompts for  image to brain or  brain to image ======================
                pure_mask_text_latents = (torch.ones((half_batch_size, seq_len), dtype=torch.long) * clip_mask_id).to(
                    accelerator.device)
                pure_mask_outputs = text_encoder(pure_mask_text_latents, return_dict=True, output_hidden_states=True)
                pure_mask_cond_embeds = (pure_mask_outputs.text_embeds).to(accelerator.device, dtype=weight_dtype)
                pure_mask_encoder_hidden_states = (pure_mask_outputs.hidden_states[-2]).to(accelerator.device,
                                                                                           dtype=weight_dtype)
                # ==============================================================================================================

            # ====================== encode masked text prompts ======================

            # ====================== fMRI perturbation  mask前半部分======================注意，fMRI在mask的时候并没有记录被mask的id，而是直接对token embedding mask
            half_batch_size, seq_len = encoding_fMRI_ids.shape
            sigma = torch.rand(half_batch_size, device=encoding_fMRI_ids.device)
            brain_mask_prob = torch.cos(sigma * math.pi * 0.5)
            brain_mask_prob = brain_mask_prob.clip(args.min_masking_rate)
            brain_timestep = brain_mask_prob.clone().clamp(min=1e-3)

            num_token_masked = (seq_len * brain_mask_prob).round().clamp(min=1)
            batch_randperm = torch.rand(half_batch_size, seq_len, device=encoding_fMRI_ids.device).argsort(dim=-1)
            mask = batch_randperm < num_token_masked.unsqueeze(-1)

            """
            model_to_call = model.module if hasattr(model, "module") else model
            if hasattr(model_to_call, "_orig_mod"):
                unwrapped_model = model_to_call._orig_mod
            else:
                unwrapped_model = model_to_call

            fmri_mask_embedding = unwrapped_model.fmri_mask_token     # shape: (D,)
            """

            # fmri_mask_embedding = torch.randn(encoding_fMRI_data.shape[2]).to(accelerator.device, dtype=weight_dtype)
            fmri_mask_embedding = torch.load(
                args.fmri_mask_token_path, map_location='cpu'
            ).to(accelerator.device, dtype=weight_dtype)


            mask_expanded = mask.unsqueeze(-1)  # broadcast (B/2, S) -> (B/2, S, 1)
            mask_token_broadcasted = fmri_mask_embedding.view(1, 1, -1).expand_as(
                encoding_fMRI_data)  # broadcast  (D,) -> (1, 1, D) -> (B/2, S, D)

            masked_brain_embs = torch.where(mask_expanded, mask_token_broadcasted, encoding_fMRI_data)
            brain_labels = torch.where(mask, encoding_fMRI_ids, -100)
            # ====================== fMRI perturbation  mask前半部分======================

            # Train Step
            img_ids = _prepare_latent_image_ids(
                masked_image_ids.shape[0],
                masked_image_ids.shape[-2],
                masked_image_ids.shape[-1],
                masked_image_ids.device,
                masked_image_ids.dtype
            )
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=masked_image_ids.device,
                                                                        dtype=masked_image_ids.dtype)
            brain_ids = prepare_brain_ids(masked_brain_embs.shape[1], device=masked_image_ids.device,
                                          dtype=masked_image_ids.dtype)

            current_mode = get_current_mode()
            print(f'当前的训练状态是：{current_mode}')

            if current_mode == "brain-image-text":
                with accelerator.accumulate(model):
                    # =================Encoding=======================
                    brain_logits = model(
                        hidden_states=image_ids,  # should be (batch size, channel, height, width)
                        encoder_hidden_states=encoder_hidden_states,  # should be (batch size, sequence_len, embed_dims)
                        thirdmodal_hidden_states=masked_brain_embs,
                        micro_conds=encoding_micro_conds,
                        pooled_projections=cond_embeds,  # should be (batch_size, projection_dim)
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=brain_mask_prob * 1000,
                    )[2]

                    brain_logits = brain_logits.reshape(-1, args.brain_vae_codebook_size)
                    # print(f"brain_logits requires_grad: {brain_logits.requires_grad}")

                    encoding_loss = F.cross_entropy(
                        brain_logits,
                        brain_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    encoding_loss = encoding_loss.reshape(half_batch_size, -1).mean(-1)
                    encoding_loss = encoding_loss / brain_timestep
                    encoding_loss = encoding_loss.mean()

                    # =================Decoding=======================
                    img_logits, text_logits, _ = model(
                        hidden_states=masked_image_ids,  # should be (batch size, channel, height, width)
                        encoder_hidden_states=masked_encoder_hidden_states,
                        # should be (batch size, sequence_len, embed_dims)
                        thirdmodal_hidden_states=decoding_fMRI_data,
                        micro_conds=decoding_micro_conds,
                        pooled_projections=masked_cond_embeds,  # should be (batch_size, projection_dim)
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=image_mask_prob * 1000,
                    )

                    img_logits = img_logits.reshape(half_batch_size, img_codebook_size, -1).permute(0, 2, 1).reshape(-1,
                                                                                                                     img_codebook_size)
                    # print(f"image_logits requires_grad: {img_logits.requires_grad}")

                    image_loss = F.cross_entropy(
                        img_logits,
                        image_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    image_loss = image_loss.reshape(half_batch_size, -1).mean(-1)
                    image_loss = image_loss / image_timestep
                    image_loss = image_loss.mean()

                    text_logits = text_logits.reshape(-1, tokenizer_vocab_size)
                    # print(f"text_logits requires_grad: {text_logits.requires_grad}")

                    text_loss = F.cross_entropy(
                        text_logits,
                        text_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    text_loss = text_loss.reshape(half_batch_size, -1).mean(-1)
                    text_loss = text_loss / text_timestep
                    text_loss = text_loss.mean()

                    decoding_loss = image_loss + text_loss

                    loss = args.encoding_loss_weight * encoding_loss + decoding_loss
                    # loss = decoding_loss

                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    avg_encoding_loss = accelerator.gather(encoding_loss.repeat(args.train_batch_size)).mean()
                    avg_decoding_loss = accelerator.gather(decoding_loss.repeat(args.train_batch_size)).mean()
                    avg_masking_rate = accelerator.gather(text_mask_prob.repeat(args.train_batch_size)).mean()

                    accelerator.backward(loss)

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            elif current_mode == "brain-image":
                # 创建 Attention Mask (屏蔽文本模态)
                image_seq_len_after_downsample = 256
                text_mask = torch.ones(half_batch_size, encoder_hidden_states.shape[1],
                                       dtype=torch.float32) * -torch.inf
                image_mask = torch.zeros(half_batch_size, image_seq_len_after_downsample, dtype=torch.float32)
                brain_mask = torch.zeros(half_batch_size, masked_brain_embs.shape[1], dtype=torch.float32)
                attention_mask = torch.cat([text_mask, image_mask, brain_mask], dim=1).unsqueeze(1).unsqueeze(1).to(
                    accelerator.device)

                with accelerator.accumulate(model):
                    # =================Encoding=======================
                    brain_logits = model(
                        hidden_states=image_ids,  # should be (batch size, channel, height, width)
                        encoder_hidden_states=pure_mask_encoder_hidden_states,
                        # should be (batch size, sequence_len, embed_dims)
                        thirdmodal_hidden_states=masked_brain_embs,
                        micro_conds=encoding_micro_conds,
                        pooled_projections=pure_mask_cond_embeds,  # should be (batch_size, projection_dim)
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=brain_mask_prob * 1000,
                        attention_mask=attention_mask,
                    )[2]

                    brain_logits = brain_logits.reshape(-1, args.brain_vae_codebook_size)

                    encoding_loss = F.cross_entropy(
                        brain_logits,
                        brain_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    encoding_loss = encoding_loss.reshape(half_batch_size, -1).mean(-1)
                    encoding_loss = encoding_loss / brain_timestep
                    encoding_loss = encoding_loss.mean()

                    # =================Decoding=======================
                    img_logits, _, _ = model(
                        hidden_states=masked_image_ids,  # should be (batch size, channel, height, width)
                        encoder_hidden_states=pure_mask_encoder_hidden_states,
                        # should be (batch size, sequence_len, embed_dims)
                        thirdmodal_hidden_states=decoding_fMRI_data,
                        micro_conds=decoding_micro_conds,
                        pooled_projections=pure_mask_cond_embeds,  # should be (batch_size, projection_dim)
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=image_mask_prob * 1000,
                        attention_mask=attention_mask,
                    )

                    img_logits = img_logits.reshape(half_batch_size, img_codebook_size, -1).permute(0, 2, 1).reshape(-1,
                                                                                                                     img_codebook_size)
                    # print(f"image_logits requires_grad: {img_logits.requires_grad}")

                    image_loss = F.cross_entropy(
                        img_logits,
                        image_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    image_loss = image_loss.reshape(half_batch_size, -1).mean(-1)
                    image_loss = image_loss / image_timestep
                    image_loss = image_loss.mean()

                    decoding_loss = image_loss

                    loss = args.encoding_loss_weight * encoding_loss + decoding_loss
                    # loss = decoding_loss

                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    avg_encoding_loss = accelerator.gather(encoding_loss.repeat(args.train_batch_size)).mean()
                    avg_decoding_loss = accelerator.gather(decoding_loss.repeat(args.train_batch_size)).mean()
                    avg_masking_rate = accelerator.gather(text_mask_prob.repeat(args.train_batch_size)).mean()

                    accelerator.backward(loss)

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            else:
                # 创建 Attention Mask (屏蔽图像模态)      pure_mask_image_latents
                image_seq_len_after_downsample = 256
                text_mask = torch.zeros(half_batch_size, encoder_hidden_states.shape[1], dtype=torch.float32)
                image_mask = torch.ones(half_batch_size, image_seq_len_after_downsample,
                                        dtype=torch.float32) * -torch.inf
                brain_mask = torch.zeros(half_batch_size, masked_brain_embs.shape[1], dtype=torch.float32)
                attention_mask = torch.cat([text_mask, image_mask, brain_mask], dim=1).unsqueeze(1).unsqueeze(1).to(
                    accelerator.device)

                with accelerator.accumulate(model):
                    # =================Encoding=======================
                    brain_logits = model(
                        hidden_states=pure_mask_image_latents,  # should be (batch size, channel, height, width)
                        encoder_hidden_states=encoder_hidden_states,  # should be (batch size, sequence_len, embed_dims)
                        thirdmodal_hidden_states=masked_brain_embs,
                        micro_conds=encoding_micro_conds,
                        pooled_projections=cond_embeds,  # should be (batch_size, projection_dim)
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=brain_mask_prob * 1000,
                        attention_mask=attention_mask,
                    )[2]

                    brain_logits = brain_logits.reshape(-1, args.brain_vae_codebook_size)
                    # print(f"brain_logits requires_grad: {brain_logits.requires_grad}")

                    encoding_loss = F.cross_entropy(
                        brain_logits,
                        brain_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    encoding_loss = encoding_loss.reshape(half_batch_size, -1).mean(-1)
                    encoding_loss = encoding_loss / brain_timestep
                    encoding_loss = encoding_loss.mean()

                    # =================Decoding=======================
                    _, text_logits, _ = model(
                        hidden_states=pure_mask_image_latents,  # should be (batch size, channel, height, width)
                        encoder_hidden_states=masked_encoder_hidden_states,
                        # should be (batch size, sequence_len, embed_dims)
                        thirdmodal_hidden_states=decoding_fMRI_data,
                        micro_conds=decoding_micro_conds,
                        pooled_projections=masked_cond_embeds,  # should be (batch_size, projection_dim)
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=image_mask_prob * 1000,
                        attention_mask=attention_mask,
                    )

                    text_logits = text_logits.reshape(-1, tokenizer_vocab_size)
                    # print(f"text_logits requires_grad: {text_logits.requires_grad}")

                    text_loss = F.cross_entropy(
                        text_logits,
                        text_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    text_loss = text_loss.reshape(half_batch_size, -1).mean(-1)
                    text_loss = text_loss / text_timestep
                    text_loss = text_loss.mean()

                    decoding_loss = text_loss

                    loss = args.encoding_loss_weight * encoding_loss + decoding_loss
                    # loss = decoding_loss

                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    avg_encoding_loss = accelerator.gather(encoding_loss.repeat(args.train_batch_size)).mean()
                    avg_decoding_loss = accelerator.gather(decoding_loss.repeat(args.train_batch_size)).mean()
                    avg_masking_rate = accelerator.gather(text_mask_prob.repeat(args.train_batch_size)).mean()

                    accelerator.backward(loss)

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if (global_step + 1) % args.logging_steps == 0:
                    logs = {
                        "current_mode": current_mode,
                        "step_loss": avg_loss.item(),
                        "encoding_loss": avg_encoding_loss.item(),
                        "decoding_loss": avg_decoding_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"current_mode: {current_mode}"
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"encoding_Loss: {avg_encoding_loss.item():0.4f} "
                        f"decoding_Loss: {avg_decoding_loss.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                if (global_step + 1) % args.checkpointing_steps == 0:
                    save_checkpoint(args, accelerator, global_step + 1, logger)

                if (global_step + 1) % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        save_embedding = fmri_mask_embedding.detach().cpu()
                        torch.save(save_embedding, os.path.join(args.output_dir, "fmri_mask_embedding.pt"))

                        save_path = os.path.join(args.output_dir, 'validation_results', f"step_{global_step + 1}")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path, exist_ok=True)

                        with torch.no_grad():
                            model.eval()

                            scheduler = Scheduler.from_pretrained(args.muddit_model_scheduler)

                            model_to_call = model.module if hasattr(model, "module") else model
                            if hasattr(model_to_call, "_orig_mod"):
                                unwrapped_model = model_to_call._orig_mod
                            else:
                                unwrapped_model = model_to_call

                            pipe = UnifiedPipeline(
                                transformer=unwrapped_model,
                                tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                vqvae=vq_model,
                                scheduler=scheduler,
                                brain_tokenizer=brain_vae
                            )

                            # --------test encoding-----------------
                            logger.info("====================Encoding: image&text to brain=========================")
                            current_fmri_mask_token_path = os.path.join(args.output_dir, "fmri_mask_embedding.pt")

                            output = pipe(
                                prompt=val_text,
                                image=val_image,
                                num_brain_token=args.num_of_brain_token,
                                height=512,
                                width=512,
                                num_inference_steps=64,
                                mask_token_embedding=args.txt_mask_token_file,
                                brain_mask_token_path=current_fmri_mask_token_path,
                                generator=torch.manual_seed(42)
                            )
                            generated_fmri_token = output.brain
                            generated_fmri = brain_vae.decoder(
                                brain_vae.post_quant_conv(generated_fmri_token.permute(0, 2, 1)))
                            recons_pcc_mm = calculate_pcc_tensor(generated_fmri, val_brain)
                            logger.info(f"Validation PCC (Img+Txt -> Brain): {recons_pcc_mm:.4f}")
                            validation_logs = {"validation_image_text_to_brain_pcc": recons_pcc_mm}

                            # validation_logs["img_recon_loss"] = img_recon_loss

                            logger.info("====================Encoding: image to brain=========================")
                            output = pipe(
                                image=val_image,
                                num_brain_token=args.num_of_brain_token,
                                height=512,
                                width=512,
                                num_inference_steps=64,
                                mask_token_embedding=args.txt_mask_token_file,
                                brain_mask_token_path=current_fmri_mask_token_path,
                                generator=torch.manual_seed(42)
                            )
                            generated_fmri_token = output.brain
                            generated_fmri = brain_vae.decoder(
                                brain_vae.post_quant_conv(generated_fmri_token.permute(0, 2, 1)))
                            recons_pcc_img = calculate_pcc_tensor(generated_fmri, val_brain)
                            logger.info(f"Validation PCC (Img -> Brain): {recons_pcc_img:.4f}")
                            validation_logs["validation_image_to_brain_pcc"] = recons_pcc_img

                            logger.info("====================Encoding: text to brain=========================")
                            output = pipe(
                                prompt=val_text,
                                num_brain_token=args.num_of_brain_token,
                                height=512,
                                width=512,
                                num_inference_steps=64,
                                mask_token_embedding=args.txt_mask_token_file,
                                brain_mask_token_path=current_fmri_mask_token_path,
                                generator=torch.manual_seed(42)
                            )
                            generated_fmri_token = output.brain
                            generated_fmri = brain_vae.decoder(
                                brain_vae.post_quant_conv(generated_fmri_token.permute(0, 2, 1)))
                            recons_pcc = calculate_pcc_tensor(generated_fmri, val_brain)
                            logger.info(f"Validation PCC (text -> Brain): {recons_pcc:.4f}")
                            validation_logs["validation_text_to_brain_pcc"] = recons_pcc

                            # --------test decoding------------------
                            logger.info("====================Decoding: brain to image&text=========================")
                            output1 = pipe(
                                brain_data=val_brain,
                                num_brain_token=args.num_of_brain_token,
                                height=512,
                                width=512,
                                num_inference_steps=64,
                                mask_token_embedding=args.txt_mask_token_file,
                                brain_mask_token_path=current_fmri_mask_token_path,
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
                            img_save_path = os.path.join(save_path, f"pcc_{recons_pcc_mm:.4f}_brain2image_text.png")
                            save_image(result, img_save_path)

                            output_data = {
                                "step": global_step + 1,
                                "prompts": generated_txt,
                            }

                            with open(os.path.join(save_path, f"brain2text_image.json"), "w") as f:
                                json.dump(output_data, f, indent=2)

                            logger.info("====================Decoding: brain to image=========================")
                            output1 = pipe(
                                brain_data=val_brain,
                                prompt=val_text,
                                num_brain_token=args.num_of_brain_token,
                                height=512,
                                width=512,
                                num_inference_steps=64,
                                mask_token_embedding=args.txt_mask_token_file,
                                brain_mask_token_path=current_fmri_mask_token_path,
                                generator=torch.manual_seed(42)
                            )
                            generated_img = output1.images

                            result = []
                            for img in generated_img:
                                if not isinstance(img, torch.Tensor):
                                    img = transforms.ToTensor()(img)
                                result.append(img.unsqueeze(0))
                            result = torch.cat(result, dim=0)
                            result = make_grid(result, nrow=3)
                            img_save_path = os.path.join(save_path, f"pcc_{recons_pcc_img:.4f}_brain2image.png")
                            save_image(result, img_save_path)

                            logger.info("====================Decoding: brain to text=========================")
                            output1 = pipe(
                                brain_data=val_brain,
                                image=val_image,
                                num_brain_token=args.num_of_brain_token,
                                height=512,
                                width=512,
                                num_inference_steps=64,
                                mask_token_embedding=args.txt_mask_token_file,
                                brain_mask_token_path=current_fmri_mask_token_path,
                                generator=torch.manual_seed(42)
                            )
                            generated_txt = output1.prompts
                            output_data = {
                                "step": global_step + 1,
                                "prompts": generated_txt,
                            }

                            with open(os.path.join(save_path, f"brain2text.json"), "w") as f:
                                json.dump(output_data, f, indent=2)

                            accelerator.log(validation_logs, step=global_step + 1)

                            model.train()
                    accelerator.wait_for_everyone()

                progress_bar.update(1)
                global_step += 1  # 必须放在if accelerator.sync_gradients:  内部，否则会在每一个minibatch被计算时都+1
            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(args, accelerator, global_step, logger)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())

"""
if __name__ == '__main__':

    CONFIG_PATH = "/nfs/diskstation/DataStation/ChangdeDu/Muddit/512/transformer/config.json"
    WEIGHTS_PATH = "/nfs/diskstation/DataStation/ChangdeDu/Muddit/512/transformer/diffusion_pytorch_model.safetensors"

    # 你的第三模态的特定参数
    THIRD_MODAL_CODEBOOK_SIZE = 128
    THIRD_MODAL_TOKEN_DIM = 16


    initialized_model = load_pretrained_weights_for_trimodal_model(
                                                                    config_path=CONFIG_PATH,
                                                                    pretrained_weights_path=WEIGHTS_PATH,
                                                                    third_modal_codebook_size=THIRD_MODAL_CODEBOOK_SIZE,
                                                                    third_modal_token_dim=THIRD_MODAL_TOKEN_DIM
                                                                    )


    print("\n模型结构:")
    print(initialized_model)


    params_to_train = freeze_pretrained_and_enable_new_modal_grads(initialized_model)
    learning_rate = 1e-4 # 示例学习率
    optimizer = torch.optim.AdamW(params_to_train, lr=learning_rate)

"""














