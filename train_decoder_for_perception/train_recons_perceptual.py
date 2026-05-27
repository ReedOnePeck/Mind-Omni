"""
训练脚本，用于训练一个从 fMRI 信号到多模态感知特征（图像和文本）的 MLP 映射器。

如何运行:
python train_recons_perceptual.py --allow_tf32
"""
from collections import defaultdict
import argparse
import logging
import math
import numpy as np
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- 导入您自己编写的模型和数据集文件 ---
# 确保 fMRI_recons_perceptual.py 和 train_perceptual_utils.py 在 Python 路径中
from fMRI_recons_perceptual import fMRI_recons_perceptron
from train_perceptual_utils import fMRI_tokenizer_TrainDataset, fMRI_tokenizer_ValDataset


# =====================================================================
# 1. 参数解析 (ArgParse)
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train an fMRI to Perceptual Features (Image & Text) Mapper")

    # --- 路径参数 ---
    parser.add_argument("--GPU_ID", type=str, default="cuda:5", )
    parser.add_argument("--output_dir", type=str,
                        default="/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/fMRI_perceptron/coarse_and_fine/",
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--fMRI_single_trial", type=str,
                        default='/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_single/',
                        help="Root directory for single-trial fMRI data.")
    parser.add_argument("--fMRI_multi_trial", type=str,
                        default='/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_multi/',
                        help="Root directory for multi-trial fMRI data.")
    parser.add_argument("--image_feature_path", type=str,
                        default='/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy',
                        help="Path to the image CLIP features (.npy).")
    parser.add_argument("--text_feature_path", type=str,
                        default='/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_feature_1024/text/text_CLIP_H_feature_1024.npy',
                        help="Path to the text CLIP features (.npy).")
    parser.add_argument("--text_hidden_feature", type=str, default="/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_H_text_max30/",
                        help="Root directory for text hidden states (.pt files).")

    # --- 训练超参数 ---
    parser.add_argument("--num_train_epochs", type=int, default=250, help="Total number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=4096, help="Batch size for the training dataloader.")
    parser.add_argument("--val_batch_size", type=int, default=512, help="Batch size for the validation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW optimizer beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW optimizer beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for the AdamW optimizer.")

    # --- 损失函数权重 ---
    parser.add_argument("--lambda_text_mse", type=float, default=0.5,
                        help="Weight for the text hidden state MSE loss component. Image MSE weight is 1.0.")

    # --- 模型架构参数 ---
    parser.add_argument("--mlp_hidden_dims", type=int, nargs='+', default=[4096, 4096, 4096, 4096],
                        help="List of hidden layer dimensions for the MLP.")
    parser.add_argument("--input_dim", type=int, default=16127, help="Input dimension of the fMRI data (N_voxel).")
    parser.add_argument("--output_dim_image", type=int, default=1024,
                        help="Output dimension of the CLIP image features.")
    parser.add_argument("--output_dim_text", type=int, default=29 * 1024,
                        help="Output dimension of the flattened CLIP text hidden states.")

    # --- 日志和保存参数 ---
    parser.add_argument("--logging_epochs", type=int, default=1, help="Log training metrics every X epochs.")
    parser.add_argument("--validation_epochs", type=int, default=1, help="Run a full validation every X epochs.")
    parser.add_argument("--checkpointing_epochs", type=int, default=10, help="Save a checkpoint every X epochs.")

    # --- 其他参数 ---
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--allow_tf32", action="store_true", help="Enable TF32 on Ampere GPUs for faster training.")

    args = parser.parse_args()
    return args


# =====================================================================
# 2. 辅助函数：PCC 计算
# =====================================================================
def calculate_pcc(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    高效地计算预测和目标之间的皮尔逊相关系数 (PCC)。
    在批次维度上计算每个样本的 PCC，然后返回平均值。
    """
    # 展平特征维度，形状变为 (B, D)
    if preds.ndim > 2: preds = preds.view(preds.shape[0], -1)
    if targets.ndim > 2: targets = targets.view(targets.shape[0], -1)

    preds_mean = preds.mean(dim=1, keepdim=True)
    targets_mean = targets.mean(dim=1, keepdim=True)

    preds_centered = preds - preds_mean
    targets_centered = targets - targets_mean

    covariance = torch.sum(preds_centered * targets_centered, dim=1)

    preds_std_dev_term = torch.sqrt(torch.sum(preds_centered ** 2, dim=1))
    targets_std_dev_term = torch.sqrt(torch.sum(targets_centered ** 2, dim=1))

    denominator = preds_std_dev_term * targets_std_dev_term

    pcc_per_sample = covariance / (denominator + 1e-8)

    return pcc_per_sample.mean()


# =====================================================================
# 3. 主训练函数
# =====================================================================
def main():
    args = parse_args()

    # --- 设置 ---
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.GPU_ID if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 日志配置 ---
    log_file_path = os.path.join(args.output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with arguments: {args}")
    logger.info(f"Using device: {device}")

    # --- 加载数据 ---
    logger.info("Loading features for datasets...")
    img_feature_full = np.load(args.image_feature_path)
    text_feature_full = np.load(args.text_feature_path)

    logger.info("Creating datasets...")
    train_dataset = fMRI_tokenizer_TrainDataset(
        fMRI_single_root=args.fMRI_single_trial,
        img_feature=img_feature_full,
        text_feature=text_feature_full,
        text_hidden_root=args.text_hidden_feature
    )
    val_dataset = fMRI_tokenizer_ValDataset(
        fMRI_single_root=args.fMRI_single_trial,
        fMRI_multi_root=args.fMRI_multi_trial,
        img_feature=img_feature_full,
        text_feature=text_feature_full,
        text_hidden_root=args.text_hidden_feature
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  num_workers=args.dataloader_num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.dataloader_num_workers)

    # --- 初始化模型和优化器 ---
    logger.info("Initializing model and optimizer...")
    model = fMRI_recons_perceptron(
        input_dim=args.input_dim,
        output_dim1=args.output_dim_image,
        output_dim2=args.output_dim_text,
        hidden_dims=args.mlp_hidden_dims
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay)

    # --- 开始训练 ---
    logger.info("***** Running training *****")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Training batch size = {args.train_batch_size}")

    for epoch in range(args.num_train_epochs):
        model.train()

        epoch_losses = {"total": [], "image_mse": [], "text_mse": []}
        epoch_pccs = {"image": [], "text": []}

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        for batch in progress_bar:
            fmri_data = batch['fmri_data'].to(device)
            image_clip_gt = batch['image_clip_feature'].to(device)
            text_hidden_gt = batch['text_hidden_state'].to(device)

            # 将 text_hidden_gt 展平以匹配模型输出
            text_hidden_gt_flat = text_hidden_gt.view(text_hidden_gt.shape[0], -1)

            # 前向传播
            pred_image_feat, pred_text_feat_flat = model(fmri_data)

            # 计算两部分的 MSE 损失
            loss_image_mse = F.mse_loss(pred_image_feat, image_clip_gt)
            loss_text_mse = F.mse_loss(pred_text_feat_flat, text_hidden_gt_flat)

            # 计算加权总损失
            total_loss = loss_image_mse + args.lambda_text_mse * loss_text_mse

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 记录 batch 指标
            epoch_losses["total"].append(total_loss.item())
            epoch_losses["image_mse"].append(loss_image_mse.item())
            epoch_losses["text_mse"].append(loss_text_mse.item())

            with torch.no_grad():
                pcc_image = calculate_pcc(pred_image_feat, image_clip_gt)
                pcc_text = calculate_pcc(pred_text_feat_flat, text_hidden_gt_flat)
                epoch_pccs["image"].append(pcc_image.item())
                epoch_pccs["text"].append(pcc_text.item())

            progress_bar.set_postfix({
                "total_loss": f"{total_loss.item():.4f}",
                "pcc_img": f"{pcc_image.item():.4f}"
            })

        # --- Epoch 级别的训练日志 ---
        if (epoch + 1) % args.logging_epochs == 0:
            avg_loss_total = np.mean(epoch_losses["total"])
            avg_loss_img = np.mean(epoch_losses["image_mse"])
            avg_loss_txt = np.mean(epoch_losses["text_mse"])
            avg_pcc_img = np.mean(epoch_pccs["image"])
            avg_pcc_txt = np.mean(epoch_pccs["text"])
            logger.info(
                f"Epoch {epoch + 1} | Train Summary: Total Loss: {avg_loss_total:.4f}, Img MSE: {avg_loss_img:.4f}, Txt MSE: {avg_loss_txt:.4f}, Img PCC: {avg_pcc_img:.4f}, Txt PCC: {avg_pcc_txt:.4f}")

        # --- 验证阶段 ---
        if (epoch + 1) % args.validation_epochs == 0:
            model.eval()

            # 初始化用于收集所有 trial 结果的容器
            val_results = defaultdict(lambda: {"losses": [], "pccs": []})

            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, desc="Validation"):
                    # 获取共享的 ground truth 特征
                    image_clip_gt_val = val_batch['image_clip_feature'].to(device)
                    text_hidden_gt_val = val_batch['text_hidden_state'].to(device)
                    text_hidden_gt_flat_val = text_hidden_gt_val.view(text_hidden_gt_val.shape[0], -1)

                    fmri_streams = {
                        "sub1_multi": val_batch['fmri_data']['subject_1'].get('multi'),
                        "sub1_single_1": val_batch['fmri_data']['subject_1'].get('single_stacked', torch.empty(0))[:, 0,
                                         :],
                        "sub1_single_2": val_batch['fmri_data']['subject_1'].get('single_stacked', torch.empty(0))[:, 1,
                                         :],
                        "sub1_single_3": val_batch['fmri_data']['subject_1'].get('single_stacked', torch.empty(0))[:, 2,
                                         :],
                        "sub5_multi": val_batch['fmri_data']['subject_5'].get('multi'),
                        "sub5_single_1": val_batch['fmri_data']['subject_5'].get('single_stacked', torch.empty(0))[:, 0,
                                         :],
                        "sub5_single_2": val_batch['fmri_data']['subject_5'].get('single_stacked', torch.empty(0))[:, 1,
                                         :],
                        "sub5_single_3": val_batch['fmri_data']['subject_5'].get('single_stacked', torch.empty(0))[:, 2,
                                         :],
                    }

                    for name, fmri_data in fmri_streams.items():
                        if fmri_data is not None and fmri_data.nelement() > 0:
                            fmri_data = fmri_data.to(device)

                            pred_img, pred_txt = model(fmri_data)

                            loss_img = F.mse_loss(pred_img, image_clip_gt_val)
                            loss_txt = F.mse_loss(pred_txt, text_hidden_gt_flat_val)
                            total_loss = loss_img + args.lambda_text_mse * loss_txt

                            pcc_img = calculate_pcc(pred_img, image_clip_gt_val)
                            pcc_txt = calculate_pcc(pred_txt, text_hidden_gt_flat_val)

                            val_results[name]["losses"].append({
                                "total": total_loss.item(),
                                "image_mse": loss_img.item(),
                                "text_mse": loss_txt.item()
                            })
                            val_results[name]["pccs"].append({
                                "image": pcc_img.item(),
                                "text": pcc_txt.item()
                            })

            # --- 打印详细的验证日志 ---
            logger.info(f"--- Epoch {epoch + 1} Validation Summary ---")
            for stream_name, results in val_results.items():
                if results["losses"]:
                    avg_total = np.mean([l["total"] for l in results["losses"]])
                    avg_img_mse = np.mean([l["image_mse"] for l in results["losses"]])
                    avg_txt_mse = np.mean([l["text_mse"] for l in results["losses"]])
                    avg_img_pcc = np.mean([p["image"] for p in results["pccs"]])
                    avg_txt_pcc = np.mean([p["text"] for p in results["pccs"]])

                    logger.info(f"  Stream: {stream_name}")
                    logger.info(
                        f"    - Avg Losses -> Total: {avg_total:.4f}, Img MSE: {avg_img_mse:.4f}, Txt MSE: {avg_txt_mse:.4f}")
                    logger.info(f"    - Avg PCC    -> Image: {avg_img_pcc:.4f}, Text: {avg_txt_pcc:.4f}")

        # --- 保存检查点 ---
        if (epoch + 1) % args.checkpointing_epochs == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
