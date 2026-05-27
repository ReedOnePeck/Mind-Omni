#!/usr/bin/env bash
set -euo pipefail

# chmod +x train_decoder_for_perception/train_recons_perceptual.sh
# ./train_decoder_for_perception/train_recons_perceptual.sh

GPU_ID="cuda:5"
OUTPUT_DIR="/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/fMRI_perceptron/coarse_and_fine/"
FMRI_SINGLE_TRIAL="/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_single/"
FMRI_MULTI_TRIAL="/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_multi/"
IMAGE_FEATURE_PATH="/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy"
TEXT_FEATURE_PATH="/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_feature_1024/text/text_CLIP_H_feature_1024.npy"
TEXT_HIDDEN_FEATURE="/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_H_text_max30/"

python train_decoder_for_perception/train_recons_perceptual.py \
    --GPU_ID "${GPU_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --fMRI_single_trial "${FMRI_SINGLE_TRIAL}" \
    --fMRI_multi_trial "${FMRI_MULTI_TRIAL}" \
    --image_feature_path "${IMAGE_FEATURE_PATH}" \
    --text_feature_path "${TEXT_FEATURE_PATH}" \
    --text_hidden_feature "${TEXT_HIDDEN_FEATURE}" \
    --train_batch_size 4096 \
    --val_batch_size 512 \
    --learning_rate 5e-4 \
    --checkpointing_epochs 10 \
    --validation_epochs 1 \
    --logging_epochs 1 \
    --allow_tf32
