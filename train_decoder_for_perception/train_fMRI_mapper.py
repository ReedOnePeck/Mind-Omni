"""

python train_fMRI_mapper.py \
    --allow_tf32
"""

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

from fMRI_MLP_mapper import fMRI_perceptron
from train_mapper_utils import fMRI_perceptron_TrainDataset, fMRI_perceptron_ValDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train an fMRI to CLIP Feature Mapper")

    # --- 路径参数 ---
    parser.add_argument("--GPU_ID", type=str, default="cuda:5", )
    parser.add_argument("--output_dir", type=str,
                        default="/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/fMRI_perceptron/",
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

    # --- 训练超参数 ---
    parser.add_argument("--num_train_epochs", type=int, default=150, help="Total number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=2048, help="Batch size for the training dataloader.")
    parser.add_argument("--val_batch_size", type=int, default=512, help="Batch size for the validation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW optimizer beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW optimizer beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for the AdamW optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for the AdamW optimizer.")

    # --- 损失函数权重 ---
    parser.add_argument("--lambda_contrastive", type=float, default=1.0,
                        help="Weight for the contrastive loss component.")
    # MSE 损失的权重默认为 1.0

    # --- 模型架构参数 ---
    parser.add_argument("--mlp_hidden_dims", type=int, nargs='+', default=[4096, 4096, 4096, 4096],
                        help="List of hidden layer dimensions for the MLP.")
    parser.add_argument("--input_dim", type=int, default=16127, help="Input dimension of the fMRI data (N_voxel).")
    parser.add_argument("--output_dim", type=int, default=1024, help="Output dimension of the CLIP features.")

    # --- 日志和保存参数 ---
    parser.add_argument("--logging_epochs", type=int, default=1, help="Log training metrics every X epochs.")
    parser.add_argument("--validation_epochs", type=int, default=1, help="Run a full validation every X epochs.")
    parser.add_argument("--checkpointing_epochs", type=int, default=10, help="Save a checkpoint every X epochs.")

    # --- 其他参数 ---
    parser.add_argument("--seed", type=int, default=20020816, help="A seed for reproducible training.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--allow_tf32", action="store_true", help="Enable TF32 on Ampere GPUs for faster training.")

    args = parser.parse_args()
    return args



def calculate_retrieval_accuracy(fmri_embeds, target_embeds, top_k_values=(1, 5, 10, 50)):
    fmri_embeds_norm = F.normalize(fmri_embeds, p=2, dim=-1);
    target_embeds_norm = F.normalize(target_embeds, p=2, dim=-1)
    sim_matrix = torch.matmul(fmri_embeds_norm, target_embeds_norm.t())
    max_k = max(top_k_values);
    _, top_k_indices = torch.topk(sim_matrix, k=max_k, dim=1)
    num_samples = fmri_embeds.shape[0];
    ground_truth = torch.arange(num_samples, device=fmri_embeds.device).view(-1, 1)
    correct_at_k = top_k_indices == ground_truth;
    accuracies = {}
    for k in top_k_values:
        num_correct = torch.any(correct_at_k[:, :k], dim=1).sum().item()
        accuracies[f"top{k}_acc"] = num_correct / num_samples
    return accuracies



def main():
    args = parse_args()


    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.GPU_ID
    os.makedirs(args.output_dir, exist_ok=True)


    log_file_path = os.path.join(args.output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode='a'), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with arguments: {args}")
    logger.info(f"Using device: {device}")

    # --- 加载数据 ---
    logger.info("Loading features for datasets...")
    img_feature_full = np.load(args.image_feature_path)
    text_feature_full = np.load(args.text_feature_path)

    logger.info("Creating datasets...")
    train_dataset = fMRI_perceptron_TrainDataset(
        fMRI_single_root=args.fMRI_single_trial,
        img_feature=img_feature_full,
        text_feature=text_feature_full
    )
    val_dataset = fMRI_perceptron_ValDataset(
        fMRI_single_root=args.fMRI_single_trial,
        fMRI_multi_root=args.fMRI_multi_trial,
        img_feature=img_feature_full,
        text_feature=text_feature_full
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  num_workers=args.dataloader_num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.dataloader_num_workers)


    logger.info("Pre-loading data for retrieval validation...")
    # 加载测试集索引
    test_indices_path = os.path.join(args.fMRI_multi_trial, 'test_data_sub1', 'test_img_index_start_from0.npy')
    test_indices = np.load(test_indices_path)

    # 加载两个被试的 multi-trial fMRI 数据
    test_fmri_sub1_path = os.path.join(args.fMRI_multi_trial, 'test_data_sub1', 'sub1_test_multi.npy')
    test_fmri_sub5_path = os.path.join(args.fMRI_multi_trial, 'test_data_sub5', 'sub5_test_multi.npy')
    test_fmri_sub1 = torch.from_numpy(np.load(test_fmri_sub1_path)).float().to(device)
    test_fmri_sub5 = torch.from_numpy(np.load(test_fmri_sub5_path)).float().to(device)

    # 根据索引筛选出对应的 CLIP 特征
    test_img_feature = torch.from_numpy(img_feature_full[test_indices]).float().to(device)
    test_text_feature = torch.from_numpy(text_feature_full[test_indices]).float().to(device)
    logger.info("Retrieval validation data loaded.")


    # --- 初始化模型和优化器 ---
    logger.info("Initializing model and optimizer...")
    model = fMRI_perceptron(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dims=args.mlp_hidden_dims
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                  eps=args.adam_epsilon, weight_decay=args.adam_weight_decay)

    logger.info("***** Running training *****")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")

    for epoch in range(args.num_train_epochs):
        model.train()
        train_loss_mse_epoch, train_loss_contrastive_epoch, train_loss_total_epoch = [], [], []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        for batch in progress_bar:
            fmri_data = batch['fmri_data'].to(device)
            image_clip = batch['image_clip_feature'].to(device)
            text_clip = batch['text_clip_feature'].to(device)

            predicted_clip = model(fmri_data)

            loss_mse = F.mse_loss(predicted_clip, image_clip)
            fmri_norm, img_norm, text_norm = [F.normalize(t, p=2, dim=-1) for t in
                                              [predicted_clip, image_clip, text_clip]]
            logits_fmri_img = torch.matmul(fmri_norm, img_norm.t())
            logits_fmri_text = torch.matmul(fmri_norm, text_norm.t())
            labels = torch.arange(fmri_data.shape[0], device=device)
            loss_contrastive = (F.cross_entropy(logits_fmri_img, labels) + F.cross_entropy(logits_fmri_text,
                                                                                           labels)) / 2
            total_loss = 0 * loss_mse + args.lambda_contrastive * loss_contrastive

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss_mse_epoch.append(loss_mse.item())
            train_loss_contrastive_epoch.append(loss_contrastive.item())
            train_loss_total_epoch.append(total_loss.item())
            progress_bar.set_postfix({"mse": f"{loss_mse.item():.4f}", "contrastive": f"{loss_contrastive.item():.4f}"})

        # --- Epoch 级别的日志 ---
        if (epoch + 1) % args.logging_epochs == 0:
            avg_mse, avg_contrastive, avg_total = [np.mean(l) for l in
                                                   [train_loss_mse_epoch, train_loss_contrastive_epoch,
                                                    train_loss_total_epoch]]
            logger.info(
                f"Epoch {epoch + 1} | Train Total Loss: {avg_total:.4f} | MSE: {avg_mse:.4f} | Contrastive: {avg_contrastive:.4f}")

        # --- 验证阶段 ---
        if (epoch + 1) % args.validation_epochs == 0:
            model.eval()

            val_fmri_embeds = {"sub1_multi": [], "sub1_single_1": [], "sub1_single_2": [], "sub1_single_3": [],
                               "sub5_multi": [], "sub5_single_1": [], "sub5_single_2": [], "sub5_single_3": []}
            val_img_clips, val_text_clips = [], []
            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, desc="Validation (Loss Calculation)"):
                    val_img_clips.append(val_batch['image_clip_feature'].to(device))
                    val_text_clips.append(val_batch['text_clip_feature'].to(device))
                    fmri_streams = {
                        "sub1_multi": val_batch['fmri_data']['subject_1']['multi'],
                        "sub1_single_1": val_batch['fmri_data']['subject_1']['single_stacked'][:, 0, :],
                        "sub1_single_2": val_batch['fmri_data']['subject_1']['single_stacked'][:, 1, :],
                        "sub1_single_3": val_batch['fmri_data']['subject_1']['single_stacked'][:, 2, :],
                        "sub5_multi": val_batch['fmri_data']['subject_5']['multi'],
                        "sub5_single_1": val_batch['fmri_data']['subject_5']['single_stacked'][:, 0, :],
                        "sub5_single_2": val_batch['fmri_data']['subject_5']['single_stacked'][:, 1, :],
                        "sub5_single_3": val_batch['fmri_data']['subject_5']['single_stacked'][:, 2, :],
                    }
                    for name, fmri_data in fmri_streams.items():
                        if fmri_data.nelement() > 0:  # 确保数据存在
                            val_fmri_embeds[name].append(model(fmri_data.to(device)))

            val_img_clips, val_text_clips = torch.cat(val_img_clips, dim=0), torch.cat(val_text_clips, dim=0)
            for name, embeds_list in val_fmri_embeds.items():
                if embeds_list: val_fmri_embeds[name] = torch.cat(embeds_list, dim=0)

            logger.info(f"--- Epoch {epoch + 1} Validation Loss Summary ---")
            for stream_name, fmri_embeds in val_fmri_embeds.items():
                if isinstance(fmri_embeds, torch.Tensor):
                    loss_mse_val = F.mse_loss(fmri_embeds, val_img_clips)
                    fmri_norm_val, img_norm_val, text_norm_val = [F.normalize(t, p=2, dim=-1) for t in
                                                                  [fmri_embeds, val_img_clips, val_text_clips]]
                    logits_img, logits_text = torch.matmul(fmri_norm_val, img_norm_val.t()), torch.matmul(fmri_norm_val,
                                                                                                          text_norm_val.t())
                    labels_val = torch.arange(fmri_embeds.shape[0], device=device)
                    loss_contrastive_val = (F.cross_entropy(logits_img, labels_val) + F.cross_entropy(logits_text,
                                                                                                      labels_val)) / 2
                    total_loss_val = loss_mse_val + args.lambda_contrastive * loss_contrastive_val
                    logger.info(
                        f"  Stream: {stream_name} | Total: {total_loss_val.item():.4f}, MSE: {loss_mse_val.item():.4f}, Contrastive: {loss_contrastive_val.item():.4f}")

            # --- 【关键修改】2. 对完整的测试集进行全局检索评估 ---
            logger.info(f"--- Epoch {epoch + 1} Validation Retrieval Summary (Full Test Set) ---")
            with torch.no_grad():
                # 使用预加载的完整测试集数据
                test_fmri_embeds_sub1 = model(test_fmri_sub1)
                test_fmri_embeds_sub5 = model(test_fmri_sub5)

            # --- 计算并记录被试 1 的检索准确率 ---
            retrieval_to_img_sub1 = calculate_retrieval_accuracy(test_fmri_embeds_sub1, test_img_feature)
            retrieval_to_text_sub1 = calculate_retrieval_accuracy(test_fmri_embeds_sub1, test_text_feature)
            logger.info("  Subject 1 (multi-trial):")
            logger.info(
                f"    - Retrieval fMRI->Image -> Top1: {retrieval_to_img_sub1['top1_acc']:.4f}, Top5: {retrieval_to_img_sub1['top5_acc']:.4f}, Top10: {retrieval_to_img_sub1['top10_acc']:.4f}, Top50: {retrieval_to_img_sub1['top50_acc']:.4f}")
            logger.info(
                f"    - Retrieval fMRI->Text  -> Top1: {retrieval_to_text_sub1['top1_acc']:.4f}, Top5: {retrieval_to_text_sub1['top5_acc']:.4f}, Top10: {retrieval_to_text_sub1['top10_acc']:.4f}, Top50: {retrieval_to_text_sub1['top50_acc']:.4f}")

            # --- 计算并记录被试 5 的检索准确率 ---
            retrieval_to_img_sub5 = calculate_retrieval_accuracy(test_fmri_embeds_sub5, test_img_feature)
            retrieval_to_text_sub5 = calculate_retrieval_accuracy(test_fmri_embeds_sub5, test_text_feature)
            logger.info("  Subject 5 (multi-trial):")
            logger.info(
                f"    - Retrieval fMRI->Image -> Top1: {retrieval_to_img_sub5['top1_acc']:.4f}, Top5: {retrieval_to_img_sub5['top5_acc']:.4f}, Top10: {retrieval_to_img_sub5['top10_acc']:.4f}, Top50: {retrieval_to_img_sub5['top50_acc']:.4f}")
            logger.info(
                f"    - Retrieval fMRI->Text  -> Top1: {retrieval_to_text_sub5['top1_acc']:.4f}, Top5: {retrieval_to_text_sub5['top5_acc']:.4f}, Top10: {retrieval_to_text_sub5['top10_acc']:.4f}, Top50: {retrieval_to_text_sub5['top50_acc']:.4f}")
            # --- 修改结束 ---

        # --- 保存检查点 ---
        if (epoch + 1) % args.checkpointing_epochs == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()