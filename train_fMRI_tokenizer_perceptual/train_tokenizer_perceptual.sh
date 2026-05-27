#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=offline

# chmod +x train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.sh
# ./train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.sh

# Experiment switches:
# 1) Leave-subject-5-out:
#    TRAIN_SUBJECTS=(1 2 3 4 6 7 8)
#    VAL_SUBJECTS=(5)
#    SUBJECT_DATA_RATIO="1.0"
#    MASK_RATIO="0.30"
#
# 2) Scaling experiments:
#    TRAIN_SUBJECTS=(1 2 3 4 5 6 7 8)
#    VAL_SUBJECTS=(1 5)
#    SUBJECT_DATA_RATIO="0.25"   # or 0.50 / 0.75
#    MASK_RATIO="0.30"
#
# 3) Text-mask-ratio ablations:
#    TRAIN_SUBJECTS=(1 2 3 4 5 6 7 8)
#    VAL_SUBJECTS=(1 5)
#    SUBJECT_DATA_RATIO="1.0"
#    MASK_RATIO="0.15"           # or 0.45 / 0.60 / 0.75

GPU_IDS="0,1"
NUM_PROCESSES=2
MAIN_PROCESS_PORT=25000
MIXED_PRECISION="bf16"

TEXT_FEATURE_1024="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_feature_1024/text/text_CLIP_H_feature_1024.npy"
IMAGE_FEATURE_1024="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy"
TEXT_HIDDEN_FEATURE="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_H_text_max30/"
FMRI_SINGLE_TRIAL="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_single/"
FMRI_MULTI_TRIAL="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/"
OUTPUT_DIR="/data/home/luyizhuo/Datastation_lyz/Models/Mind_omni_rebuttal/tokenizer/"

TRAIN_SUBJECTS=(1 2 3 4 6 7 8)
VAL_SUBJECTS=(1 5)
SUBJECT_DATA_RATIO="0.5"
MASK_RATIO="0.30"

PYTHON_PATH="./" accelerate launch \
    --multi_gpu \
    --gpu_ids "${GPU_IDS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --num_processes "${NUM_PROCESSES}" \
    train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.py \
    --text_feature_1024 "${TEXT_FEATURE_1024}" \
    --image_feature_1024 "${IMAGE_FEATURE_1024}" \
    --text_hidden_feature "${TEXT_HIDDEN_FEATURE}" \
    --fMRI_single_trial "${FMRI_SINGLE_TRIAL}" \
    --fMRI_multi_trial "${FMRI_MULTI_TRIAL}" \
    --train_batch_size 224 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_grad_norm 10 \
    --lambda_mse 1.0 \
    --lambda_commitment 0.8 \
    --lambda_contrastive 0.08 \
    --lambda_distillation 0.6 \
    --lambda_fine_grained 0.02 \
    --lambda_txt_perceptual_loss 0.5 \
    --lambda_img_perceptual_loss 0.5 \
    --num_res_blocks 3 \
    --codebook_size 128 \
    --codebook_embed_dim 16 \
    --desired_token_num 64 \
    --train_subjects "${TRAIN_SUBJECTS[@]}" \
    --val_subjects "${VAL_SUBJECTS[@]}" \
    --subject_data_ratio "${SUBJECT_DATA_RATIO}" \
    --mask_ratio "${MASK_RATIO}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 300 \
    --use_8bit_adam \
    --allow_tf32 \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers 4 \
    --max_train_steps 10002 \
    --checkpointing_steps 2000 \
    --logging_steps 50 \
    --checkpoints_total_limit 20 \
    --retrieval_validation_steps 2000 \
    --validation_epochs 1 \
    --report_to "wandb"
