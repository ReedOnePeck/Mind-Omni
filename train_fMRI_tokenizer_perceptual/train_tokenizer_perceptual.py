import argparse
import copy
import logging
import math
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append(os.getcwd())
import json
from dataclasses import asdict, is_dataclass
import torch
import torch.nn.functional as F
from torch import nn

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from torch.utils.data import DataLoader
from torchvision import transforms

import diffusers.optimization
from MindOmni_utils.scheduler import Scheduler
from diffusers.utils import is_wandb_available
from MindOmni_utils.trainer_utils import save_checkpoint

from train_fMRI_tokenizer_perceptual.fMRI_tokenizer_perceptual import VQ_fMRI
from train_fMRI_tokenizer_perceptual.dataset_utils import fMRI_tokenizer_TrainDataset, fMRI_tokenizer_ValDataset
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")

import torch._dynamo
torch._dynamo.config.verbose = True

# Optionally suppress errors to fall back to eager execution
torch._dynamo.config.suppress_errors = False


from accelerate.utils import DistributedDataParallelKwargs



def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for fMRI Tokenizer Training")
    # ========================
    # General Model Parameters
    # ========================
    parser.add_argument("--n_voxel", type=int, default=16127,
                        help="fMRI data original length.")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Base channels for the first convolution in encoder/decoder.")
    parser.add_argument("--z_channels", type=int, default=256,
                        help="Latent dimension (after encoder, before quantization).")
    parser.add_argument("--desired_token_num", type=int, default=64,
                        help="number of fMRI tokens，32 or 64")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="Number of residual blocks in encoder/decoder.")
    parser.add_argument("--dropout_p", type=float, default=0.0,
                        help="Dropout probability.")

    # ========================
    # Codebook Parameters
    # ========================
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Number of entries in the codebook. Recommended: 1024 - 4096.")
    parser.add_argument("--codebook_embed_dim", type=int, default=256,
                        help="Dimension of the codebook embeddings. Recommended: 256 or 512.")
    # For boolean flags, it's common to use `store_true` or `store_false`.
    # If the flag is present, it sets the value to True.
    parser.add_argument("--codebook_l2_norm", action='store_true', default=True,
                        help="Apply L2 normalization to codebook embeddings. Default is True.")
    parser.add_argument("--no_codebook_l2_norm", action='store_false', dest='codebook_l2_norm',
                        help="Disable L2 normalization for codebook embeddings.")
    parser.add_argument("--codebook_show_usage", action='store_true', default=True,
                        help="Show codebook usage during training. Default is True.")
    parser.add_argument("--no_codebook_show_usage", action='store_false', dest='codebook_show_usage',
                        help="Disable showing codebook usage during training.")

    # ========================
    # Channel Multipliers                                      暂定64个fMRI token
    # ========================
    parser.add_argument("--encoder_ch_mult", type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help="Channel multipliers for the encoder blocks. e.g., --encoder_ch_mult 1 2 4 8 16")
    parser.add_argument("--decoder_ch_mult", type=int, nargs='+', default=[16, 8, 4, 2, 1],
                        help="Channel multipliers for the decoder blocks. e.g., --decoder_ch_mult 16 8 4 2 1")

    # ========================
    # Loss Weights & Parameters
    # ========================
    parser.add_argument("--entropy_loss_ratio", type=float, default=0.0,
                        help="Weight for the entropy loss.")
    parser.add_argument("--lambda_mse", type=float, default=1.0,
                        help="Weight for the Mean Squared Error (reconstruction) loss.")
    parser.add_argument("--lambda_commitment", type=float, default=0.25,
                        help="Weight for the codebook commitment loss (beta).")
    parser.add_argument("--lambda_contrastive", type=float, default=1.0,
                        help="Weight for the contrastive loss.")
    parser.add_argument("--lambda_distillation", type=float, default=0.5,
                        help="Weight for the distillation loss.")
    parser.add_argument("--lambda_fine_grained", type=float, default=1.0,
                        help="Weight for the fine-grained loss.")
    parser.add_argument("--lambda_txt_perceptual_loss", type=float, default=0.5,
                        help="Weight for the txt_perceptual_loss.")
    parser.add_argument("--lambda_img_perceptual_loss", type=float, default=0.5,
                        help="Weight for the img_perceptual_loss.")


    # ========================
    # Masked Language Modeling (MLM) Parameters
    # ========================
    parser.add_argument("--mask_ratio", type=float, default=0.30,
                        help="Masking rate for text tokens in MLM.")
    parser.add_argument("--mlm_temp", type=float, default=1.0,
                        help="Temperature for the MLM softmax.")
    parser.add_argument("--clip_sos_token_id", type=int, default=49406,
                        help="The [SOS] token ID of the CLIP tokenizer.")

    # Example of the pre-existing argument for context
    parser.add_argument(
        "--text_encoder_architecture",
        type=str,
        default="open_clip",
        required=False,
        help="The architecture of the text encoder. One of ['CLIP', 'open_clip', 'flan-t5-base','Qwen2-0.5B','gemini-2b',long_t5_clip','t5_clip']",
    )
    parser.add_argument(
        "--text_feature_1024",
        type=str,
        default='/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_feature_1024/text/text_CLIP_H_feature_1024.npy',
        required=False,
        help="text feature extracted by CLIP_H,shape:(73000,1024)",
    )
    parser.add_argument(
        "--image_feature_1024",
        type=str,
        default='/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy',
        required=False,
        help="image feature extracted by CLIP_H,shape:(73000,1024)",
    )
    parser.add_argument(
        "--text_hidden_feature",
        type=str,
        default='/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_H_text_max30/',
        required=False,
        help="A folder for text hidden states(-2) from CLIP_H;   Name:from 00000.pt to 72999.pt)"
             "data_dict = torch.load(file_path);data_dict['input_ids']  Tensor (30,), data_dict['attention_mask']  Tensor (30,),  data_dict['last_hidden_state']  Tensor (30, 1024)"
    )
    parser.add_argument(
        "--fMRI_single_trial",
        type=str,
        default='/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_single/',
        required=False,
    )
    parser.add_argument(
        "--fMRI_multi_trial",
        type=str,
        default='/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/',
        required=False,
    )
    parser.add_argument(
        "--train_subjects",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Subject ids used for training.",
    )
    parser.add_argument(
        "--val_subjects",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Subject ids used for validation and retrieval evaluation.",
    )
    parser.add_argument(
        "--subject_data_ratio",
        type=float,
        default=1.0,
        help="Fraction of training data kept for each training subject. Must be in (0, 1].",
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
        default="/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/fMRI_tokenizer/",
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
        "--retrieval_validation_steps",
        type=int,
        default=200,
        help=(
            "Run retrieval every X steps. chance level:1/1000"
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help=(
            "Run validation every X epochs."
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
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.", required=False)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    args = parser.parse_args()

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    return args


@dataclass
class ModelArgs:
    """
    一个专门用于 VQ_fMRI 模型架构的配置类。
    """
    # General Model Parameters
    n_voxel: int = 16127
    base_channels: int = 64
    z_channels: int = 256
    num_res_blocks: int = 2
    dropout_p: float = 0.0
    desired_token_num: int = 64    #只能取32或者64

    # Codebook Parameters
    codebook_size: int = 1024
    codebook_embed_dim: int = 256
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True

    # Channel Multipliers
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [16, 8, 4, 2, 1])

    # Loss Weights & Parameters (这些也可以放在模型配置中，如果模型内部计算损失)
    entropy_loss_ratio: float = 0.0
    lambda_mse: float = 1.0
    lambda_commitment: float = 0.25
    lambda_contrastive: float = 1.0
    lambda_distillation: float = 0.5
    lambda_fine_grained: float = 1.0
    lambda_txt_perceptual_loss: float = 0.5
    lambda_img_perceptual_loss: float = 0.5

    # MLM Parameters
    mask_ratio: float = 0.30
    mlm_temp: float = 1.0
    clip_sos_token_id: int = 49406


def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    print("ddp_kwargs:",  ddp_kwargs)
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
    #logging.basicConfig(
    #    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #    datefmt="%m/%d/%Y %H:%M:%S",
    #    level=logging.INFO,
    #)

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
        log_file_path = os.path.join(args.output_dir, "training_log.txt")
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
        accelerator.init_trackers("MindOmni_fMRI_tokenizer", config=vars(copy.deepcopy(args)))

    if args.seed is not None:
        set_seed(args.seed)

    model_config = ModelArgs(
        n_voxel=args.n_voxel,
        base_channels=args.base_channels,
        z_channels=args.z_channels,
        desired_token_num = args.desired_token_num,
        num_res_blocks=args.num_res_blocks,
        dropout_p=args.dropout_p,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        codebook_show_usage=args.codebook_show_usage,
        encoder_ch_mult=args.encoder_ch_mult,
        decoder_ch_mult=args.decoder_ch_mult,
        entropy_loss_ratio=args.entropy_loss_ratio,
        lambda_mse=args.lambda_mse,
        lambda_commitment=args.lambda_commitment,
        lambda_contrastive=args.lambda_contrastive,
        lambda_distillation=args.lambda_distillation,
        lambda_fine_grained=args.lambda_fine_grained,
        lambda_txt_perceptual_loss=args.lambda_txt_perceptual_loss,
        lambda_img_perceptual_loss=args.lambda_img_perceptual_loss,
        mask_ratio=args.mask_ratio,
        mlm_temp=args.mlm_temp,
        clip_sos_token_id=args.clip_sos_token_id,
    )

    model = VQ_fMRI(model_config)
    model = torch.compile(model)
    model.train()
    #model.requires_grad_(True)
    """
    for param in model.fMRI_perceptron.parameters():
        param.requires_grad = False
    """

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    def save_model_hook(models, weights, output_dir):
        """
        处理自定义的 nn.Module。它会保存模型的 state_dict 和 config.json。
        """
        if accelerator.is_main_process:
            if len(models) > 0:
                # 获取被 accelerator 包装的原始模型对象
                model_to_save = accelerator.unwrap_model(models[0])

                # 定义保存模型的子目录
                model_output_dir = os.path.join(output_dir, "VQ_fMRI")
                os.makedirs(model_output_dir, exist_ok=True)

                # 1. 保存模型的 state_dict (权重)
                torch.save(model_to_save.state_dict(), os.path.join(model_output_dir, "pytorch_model.bin"))

                # 2. 保存模型的 config (配置)
                #    我们假设模型的 config 是一个 dataclass
                if hasattr(model_to_save, 'config') and is_dataclass(model_to_save.config):
                    # 将 dataclass 转换为可以序列化为 JSON 的字典
                    config_dict = asdict(model_to_save.config)
                    with open(os.path.join(model_output_dir, "config.json"), 'w') as f:
                        json.dump(config_dict, f, indent=2)

                print(f"Custom model state_dict and config saved to {model_output_dir}")

            # 弹出权重，告知 accelerator 已处理
            while len(weights) > 0:
                weights.pop()

    def load_model_hook(models, input_dir):
        """
        处理自定义的 nn.Module，并兼容 torch.compile()。它会先加载 config.json 来实例化模型，然后加载 state_dict。
        """

        # --- 兼容 torch.compile() 的辅助函数 ---
        def adapt_state_dict_for_compile(state_dict):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict["_orig_mod." + key] = value
            return new_state_dict

        if len(models) > 0:
            model_to_load_into = models.pop()

            # 定义加载模型的子目录
            model_input_dir = os.path.join(input_dir, "VQ_fMRI")

            # 1. 加载模型的 config.json
            config_path = os.path.join(model_input_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}. Cannot recreate model.")

            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # 2. 使用加载的配置来实例化一个新的模型
            #    这个模型仅用于加载权重，不会被训练
            #    你需要确保 ModelArgs 能够接受字典作为输入，dataclass 默认支持
            loaded_model_config = ModelArgs(**config_dict)
            # 假设你的 VQ_fMRI 类可以这样被实例化
            temp_model_for_loading = VQ_fMRI(config=loaded_model_config)

            # 3. 加载模型的 state_dict
            weights_path = os.path.join(model_input_dir, "pytorch_model.bin")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found at {weights_path}.")

            loaded_state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

            # 4. 加载 state_dict 到临时模型中 (可选但推荐，用于检查键是否匹配)
            temp_model_for_loading.load_state_dict(loaded_state_dict)

            # 5. 调整 state_dict 以兼容 torch.compile()
            adapted_state_dict = adapt_state_dict_for_compile(temp_model_for_loading.state_dict())

            # 6. 将最终的 state_dict 加载到 accelerator 管理的模型中
            model_to_load_into.load_state_dict(adapted_state_dict)

            print(f"Custom model loaded from {model_input_dir} with compile compatibility.")

            # 释放临时模型的内存
            del temp_model_for_loading




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
            "params": [p for p in model.parameters() if p.requires_grad],
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
    logger.info(f"Training subjects: {args.train_subjects}")
    logger.info(f"Validation subjects: {args.val_subjects}")
    logger.info(f"Per-subject training data ratio: {args.subject_data_ratio}")

    print(f"Loading image features from {args.image_feature_1024}...")
    img_feature = np.load(args.image_feature_1024)
    print(f"Loading text features from {args.text_feature_1024}...")
    text_feature = np.load(args.text_feature_1024)




    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    train_dataset = fMRI_tokenizer_TrainDataset(
        fMRI_single_root=args.fMRI_single_trial,
        fMRI_multi_root=args.fMRI_multi_trial,
        img_feature=img_feature,
        text_feature=text_feature,
        text_hidden_root=args.text_hidden_feature,
        train_subjects=args.train_subjects,
        subject_data_ratio=args.subject_data_ratio,
        seed=args.seed,
    )
    val_dataset = fMRI_tokenizer_ValDataset(
        fMRI_single_root=args.fMRI_single_trial,
        fMRI_multi_root=args.fMRI_multi_trial,
        img_feature=img_feature,
        text_feature=text_feature,
        text_hidden_root=args.text_hidden_feature,
        subjects_to_use=args.val_subjects,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    train_dataloader.num_batches = len(train_dataloader)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    val_dataloader.num_batches = len(val_dataloader)

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )

    logger.info("Preparing model, optimizer and dataloaders")

    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    train_dataloader.num_batches = len(train_dataloader)
    val_dataloader.num_batches = len(val_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16




    test_fmri_by_subject = {}
    retrieval_indices = None
    for subject_id in args.val_subjects:
        print(f"Loading sub{subject_id} test fMRI")
        multi_subject_dir = os.path.join(args.fMRI_multi_trial, f"test_data_sub{subject_id}")
        multi_subject_path = os.path.join(multi_subject_dir, f"sub{subject_id}_test_multi.npy")
        test_fmri_by_subject[subject_id] = torch.from_numpy(np.load(multi_subject_path)).float().to(accelerator.device)

        if retrieval_indices is None:
            retrieval_indices = np.load(os.path.join(multi_subject_dir, "test_img_index_start_from0.npy"))

    test_img_feature = torch.from_numpy(img_feature[retrieval_indices]).float().to(accelerator.device)
    test_text_feature = torch.from_numpy(text_feature[retrieval_indices]).float().to(accelerator.device)






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
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(os.path.basename(resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch


    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        progress_bar = tqdm(
            range(num_update_steps_per_epoch),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{num_train_epochs}"
        )


        for batch in train_dataloader:
            torch.cuda.empty_cache()
            fmri_data = batch['fmri_data'].to(accelerator.device, dtype=weight_dtype)
            image_clip_feature = batch['image_clip_feature'].to(accelerator.device, dtype=weight_dtype)
            text_clip_feature = batch['text_clip_feature'].to(accelerator.device, dtype=weight_dtype)
            input_ids = batch['text_input_ids'].to(accelerator.device, dtype=torch.int64)
            attention_mask = batch['text_attention_mask'].to(accelerator.device, dtype=torch.int64)
            text_hidden_state = batch['text_hidden_state'].to(accelerator.device, dtype=weight_dtype)

            # Train Step
            with accelerator.accumulate(model):
                train_total_loss = model(fmri_data=fmri_data, img_clip_feature=image_clip_feature, text_clip_feature=text_clip_feature,
                                                     text_clip_hidden_features=text_hidden_state, text_input_ids=input_ids, text_padding_mask=attention_mask)
                accelerator.backward(train_total_loss)

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)


            if accelerator.sync_gradients:
                if (global_step + 1) % args.logging_steps == 0:
                    with torch.no_grad():

                        model_to_call = model.module if hasattr(model, "module") else model
                        if hasattr(model_to_call, "_orig_mod"):
                            unwrapped_model = model_to_call._orig_mod
                        else:
                            unwrapped_model = model_to_call

                        mixed_precision_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16

                        with torch.autocast(device_type=accelerator.device.type, dtype=mixed_precision_dtype):
                            # 在这个上下文中，对 unwrapped_model (float32) 的调用能够正确处理 bfloat16 的输入数据
                            unwrapped_model.eval()# 切换到评估模式，获取完整的损失字典
                            Train_loss = unwrapped_model(fmri_data=fmri_data, img_clip_feature=image_clip_feature,
                                                           text_clip_feature=text_clip_feature,
                                                           text_clip_hidden_features=text_hidden_state, text_input_ids=input_ids,
                                                           text_padding_mask=attention_mask, return_loss_dict=True)
                            unwrapped_model.train()  # 别忘了切换回训练模式

                    avg_train_recons_pcc = accelerator.gather(
                        Train_loss["recons_pcc"].repeat(args.train_batch_size)).mean()
                    avg_train_total_loss = accelerator.gather(
                        Train_loss["total_loss"].repeat(args.train_batch_size)).mean()
                    avg_train_recon_loss = accelerator.gather(
                        Train_loss["recon_loss"].repeat(args.train_batch_size)).mean()
                    avg_train_commit_loss = accelerator.gather(
                        Train_loss["commit_loss"].repeat(args.train_batch_size)).mean()
                    avg_train_contrastive_loss = accelerator.gather(
                        Train_loss["contrastive_loss"].repeat(args.train_batch_size)).mean()
                    avg_train_distillation_loss = accelerator.gather(
                        Train_loss["distillation_loss"].repeat(args.train_batch_size)).mean()
                    avg_train_fine_grained_loss = accelerator.gather(
                        Train_loss["fine_grained_loss"].repeat(args.train_batch_size)).mean()

                    avg_train_img_perceptual_loss = accelerator.gather(
                        Train_loss["img_perceptual_loss"].repeat(args.train_batch_size)).mean()
                    avg_train_txt_perceptual_loss = accelerator.gather(
                        Train_loss["txt_perceptual_loss"].repeat(args.train_batch_size)).mean()
                    avg_train_perceptual_img_pcc = accelerator.gather(
                        Train_loss["perceptual_img_pcc"].repeat(args.train_batch_size)).mean()
                    avg_train_perceptual_txt_pcc = accelerator.gather(
                        Train_loss["perceptual_txt_pcc"].repeat(args.train_batch_size)).mean()

                    avg_train_codebook_usage = accelerator.gather(
                        torch.tensor(Train_loss["codebook_usage"], device=accelerator.device).repeat(
                            args.train_batch_size)).mean()


                    logs = {
                        "step_train_total_loss": avg_train_total_loss.item(),
                        "step_train_recon_loss": avg_train_recon_loss.item(),
                        "step_train_commit_loss": avg_train_commit_loss.item(),
                        "step_train_contrastive_loss": avg_train_contrastive_loss.item(),
                        "step_train_distillation_loss": avg_train_distillation_loss.item(),
                        "step_train_fine_grained_loss": avg_train_fine_grained_loss.item(),
                        "step_train_codebook_usage": avg_train_codebook_usage.item(),
                        "step_train_recons_pcc": avg_train_recons_pcc.item(),

                        "step_train_img_perceptual_loss": avg_train_img_perceptual_loss.item(),
                        "step_train_txt_perceptual_loss": avg_train_txt_perceptual_loss.item(),
                        "step_train_perceptual_img_pcc": avg_train_perceptual_img_pcc.item(),
                        "step_train_perceptual_txt_pcc": avg_train_perceptual_txt_pcc.item(),

                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    accelerator.log(logs, step=global_step + 1)

                    log_message = (
                        f"Step: {global_step + 1} | "
                        f"Total Loss: {avg_train_total_loss.item():.4f} | "
                        f"Recons PCC: {avg_train_recons_pcc.item():.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.6f}\n"
                        f"  └─ Details: Recon: {avg_train_recon_loss.item():.4f}, "
                        f"Commit: {avg_train_commit_loss.item():.4f}, "
                        f"Contrastive: {avg_train_contrastive_loss.item():.4f}, "
                        f"Distill: {avg_train_distillation_loss.item():.4f}, "
                        f"FineGrained: {avg_train_fine_grained_loss.item():.4f}\n"
                        f"  └─ Perceptual: ImgLoss: {avg_train_img_perceptual_loss.item():.4f}, "
                        f"TxtLoss: {avg_train_txt_perceptual_loss.item():.4f}, "
                        f"ImgPCC: {avg_train_perceptual_img_pcc.item():.4f}, "
                        f"TxtPCC: {avg_train_perceptual_txt_pcc.item():.4f}\n"
                        f"  └─ Codebook Usage: {avg_train_codebook_usage.item():.2f}"
                    )
                    logger.info(log_message)



                if (global_step + 1) % args.checkpointing_steps == 0:
                    save_checkpoint(args, accelerator, global_step + 1, logger)



                if (global_step + 1) % args.retrieval_validation_steps == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        logger.info("Generating images...")



                        # --- 关键修改：健壮的手动解包逻辑 ---

                        # 1. 首先，剥开 DDP 层（如果存在）。
                        #    在多卡 DDP 环境下，model.module 指向被 DDP 包装的对象。
                        #    在单卡环境下，model 没有 .module 属性，所以我们需要检查。
                        model_to_call = model.module if hasattr(model, "module") else model

                        # 2. 然后，检查 torch.compile 的包装层。
                        #    torch.compile 成功后，会在对象上添加一个 _orig_mod 属性。
                        if hasattr(model_to_call, "_orig_mod"):
                            # 如果模型已经被编译，我们就获取最内层的原始模型
                            unwrapped_model = model_to_call._orig_mod
                        else:
                            # 如果还没被编译（比如第一次迭代），那么剥开 DDP 后的对象就是我们需要的
                            unwrapped_model = model_to_call

                        unwrapped_model.eval()
                        retrieval_logs = {}
                        retrieval_lines = [f"Step: {global_step + 1} | --- Validation Acc ---"]
                        for subject_id in args.val_subjects:
                            retrieval_results = unwrapped_model.calculate_retrieval(
                                fmri_data=test_fmri_by_subject[subject_id],
                                img_clip_feature=test_img_feature,
                                text_clip_feature=test_text_feature,
                            )
                            retrieval_logs[f"step_val_sub{subject_id}_f2i_top1acc"] = retrieval_results['fmri_to_image_acc'].item()
                            retrieval_logs[f"step_val_sub{subject_id}_f2t_top1acc"] = retrieval_results['fmri_to_text_acc'].item()
                            retrieval_lines.append(
                                f"  Sub{subject_id}: fMRI-to-Image = {retrieval_results['fmri_to_image_acc']:.4f}, "
                                f"fMRI-to-Text = {retrieval_results['fmri_to_text_acc']:.4f}"
                            )
                        unwrapped_model.train()

                    accelerator.log(retrieval_logs, step=global_step + 1)
                    logger.info("\n".join(retrieval_lines))

                progress_bar.update(1)
                global_step += 1    #必须放在if accelerator.sync_gradients:  内部，否则会在每一个minibatch被计算时都+1

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

        if (epoch + 1) % args.validation_epochs == 0:
            logger.info(f"--- Running full validation for Epoch {epoch + 1} ---")
            model.eval()

            val_results = {}
            for subject_id in args.val_subjects:
                val_results[f"sub{subject_id}_multi"] = {"losses": [], "pccs": [], "img_pccs": [], "txt_pccs": []}
                for single_idx in [1, 2, 3]:
                    val_results[f"sub{subject_id}_single_{single_idx}"] = {"losses": [], "pccs": []}

            with torch.no_grad():

                model_to_call = model.module if hasattr(model, "module") else model
                if hasattr(model_to_call, "_orig_mod"):
                    unwrapped_model = model_to_call._orig_mod
                else:
                    unwrapped_model = model_to_call

                mixed_precision_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16


                for val_batch in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
                    image_clip_feature = val_batch['image_clip_feature'].to(accelerator.device, dtype=weight_dtype)
                    text_clip_feature = val_batch['text_clip_feature'].to(accelerator.device, dtype=weight_dtype)
                    input_ids = val_batch['text_input_ids'].to(accelerator.device, dtype=torch.int64)
                    attention_mask = val_batch['text_attention_mask'].to(accelerator.device, dtype=torch.int64)
                    text_hidden_state = val_batch['text_hidden_state'].to(accelerator.device, dtype=weight_dtype)

                    # 获取当前局部批次的大小，用于 gather
                    local_batch_size = image_clip_feature.shape[0]

                    fmri_streams = {}
                    for subject_id in args.val_subjects:
                        subject_key = f"subject_{subject_id}"
                        subject_fmri = val_batch['fmri_data'][subject_key]
                        fmri_streams[f"sub{subject_id}_multi"] = subject_fmri['multi'].to(accelerator.device, dtype=weight_dtype)
                        for single_idx in [1, 2, 3]:
                            fmri_streams[f"sub{subject_id}_single_{single_idx}"] = (
                                subject_fmri['single_stacked'][:, single_idx - 1, :].to(accelerator.device, dtype=weight_dtype)
                            )

                    for stream_name, fmri_data in fmri_streams.items():
                        with torch.autocast(device_type=accelerator.device.type, dtype=mixed_precision_dtype):
                            val_loss_dict = unwrapped_model(
                                                fmri_data=fmri_data,
                                                img_clip_feature=image_clip_feature,
                                                text_clip_feature=text_clip_feature,
                                                text_clip_hidden_features=text_hidden_state,
                                                text_input_ids=input_ids,
                                                text_padding_mask=attention_mask,
                                                return_loss_dict = True
                                            )

                        val_total_loss = val_loss_dict["total_loss"]
                        val_pcc = val_loss_dict["recons_pcc"]

                        avg_batch_loss = accelerator.gather(val_total_loss.repeat(local_batch_size)).mean()
                        avg_batch_pcc = accelerator.gather(val_pcc.repeat(local_batch_size)).mean()

                        val_results[stream_name]["losses"].append(avg_batch_loss)
                        val_results[stream_name]["pccs"].append(avg_batch_pcc)

                        # 为multi流添加ImgPCC和TxtPCC
                        if "multi" in stream_name:
                            val_img_pcc = val_loss_dict["perceptual_img_pcc"]
                            val_txt_pcc = val_loss_dict["perceptual_txt_pcc"]

                            avg_batch_img_pcc = accelerator.gather(val_img_pcc.repeat(local_batch_size)).mean()
                            avg_batch_txt_pcc = accelerator.gather(val_txt_pcc.repeat(local_batch_size)).mean()

                            val_results[stream_name]["img_pccs"].append(avg_batch_img_pcc)
                            val_results[stream_name]["txt_pccs"].append(avg_batch_txt_pcc)


            # 这个计算和日志记录只在主进程上进行，避免重复
            if accelerator.is_main_process:
                # 用于发送到 W&B / TensorBoard 的日志字典
                final_epoch_logs = {}
                # 用于打印到终端的日志消息列表
                final_log_messages = [f"--- Epoch {epoch + 1} Validation Summary ---"]

                # 遍历每个数据流的收集结果
                for stream_name, results in val_results.items():
                    # 计算整个 epoch 的平均 loss 和 pcc
                    # torch.stack 将列表转换为张量，然后求平均
                    if results["losses"]:  # 确保列表不为空
                        epoch_avg_loss = torch.stack(results["losses"]).mean().item()
                        epoch_avg_pcc = torch.stack(results["pccs"]).mean().item()

                        # 填充日志字典
                        final_epoch_logs[f"epoch_val_{stream_name}_loss"] = epoch_avg_loss
                        final_epoch_logs[f"epoch_val_{stream_name}_pcc"] = epoch_avg_pcc


                        # 为multi流添加ImgPCC和TxtPCC
                        if "multi" in stream_name:
                            epoch_avg_img_pcc = torch.stack(results["img_pccs"]).mean().item()
                            epoch_avg_txt_pcc = torch.stack(results["txt_pccs"]).mean().item()

                            final_epoch_logs[f"epoch_val_{stream_name}_img_pcc"] = epoch_avg_img_pcc
                            final_epoch_logs[f"epoch_val_{stream_name}_txt_pcc"] = epoch_avg_txt_pcc

                            # 构造包含ImgPCC和TxtPCC的日志消息
                            final_log_messages.append(
                                f"  - {stream_name}: Avg Loss = {epoch_avg_loss:.4f}, Avg PCC = {epoch_avg_pcc:.4f}, "
                                f"ImgPCC = {epoch_avg_img_pcc:.4f}, TxtPCC = {epoch_avg_txt_pcc:.4f}"
                            )
                        else:
                            # 对于非multi流，保持原来的日志格式
                            final_log_messages.append(
                                f"  - {stream_name}: Avg Loss = {epoch_avg_loss:.4f}, Avg PCC = {epoch_avg_pcc:.4f}"
                            )

                # 将所有 epoch 级别的指标，在当前的 global_step 处记录下来
                accelerator.log(final_epoch_logs, step=global_step)

                # 将所有日志消息合并成一个字符串并打印
                logger.info("\n".join(final_log_messages))

            # 评估结束后，将模型切回训练模式
            model.train()




    #End epoch
    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(args, accelerator, global_step, logger)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model_to_call = model.module if hasattr(model, "module") else model
        if hasattr(model_to_call, "_orig_mod"):
            unwrapped_model = model_to_call._orig_mod
        else:
            unwrapped_model = model_to_call
        unwrapped_model.save_pretrained(args.output_dir)

    accelerator.end_training()



if __name__ == "__main__":
    main(parse_args())











