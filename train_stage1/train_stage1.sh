#!/usr/bin/env bash
set -euo pipefail

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline

# chmod +x train_stage1/train_stage1.sh
# ./train_stage1/train_stage1.sh

GPU_IDS="1,2,3,4"
NUM_PROCESSES=4
MAIN_PROCESS_PORT=25000
MIXED_PRECISION="bf16"

MUDDIT_MODEL_CONFIG="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/512/transformer/config.json"
MUDDIT_MODEL_CKPT="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/512/transformer/diffusion_pytorch_model.safetensors"
TXT_MASK_TOKEN_FILE="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth"
BRAIN_VAE_MODEL_CKPT="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI"
FMRI_SINGLE_TRIAL="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_single/"
IMG_TOKEN_IDS="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/VQVAE_feature_img/"
TXT_TOKEN_IDS="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/caption_ids_COCO_recaption/"
OUTPUT_DIR="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_with_encoding/"
FMRI_MASK_TOKEN_PATH="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_with_encoding/fmri_mask_embedding.pt"
RESUME_FROM_CHECKPOINT="latest"

VAL_IMGS=(
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00000.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00100.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00200.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00300.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00400.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00500.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00600.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00700.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00800.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/00900.png"
)

VAL_TEXT=(
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00000.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00100.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00200.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00300.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00400.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00500.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00600.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00700.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00800.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/00900.txt"
)

VAL_BRAIN=(
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00000.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00100.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00200.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00300.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00400.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00500.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00600.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00700.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00800.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/00900.npy"
)

PYTHON_PATH="./" accelerate launch \
    --multi_gpu \
    --gpu_ids "${GPU_IDS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --num_processes "${NUM_PROCESSES}" \
    train_stage1/train_mind_omni_stage1.py \
    --muddit_model_config "${MUDDIT_MODEL_CONFIG}" \
    --muddit_model_ckpt "${MUDDIT_MODEL_CKPT}" \
    --txt_mask_token_file "${TXT_MASK_TOKEN_FILE}" \
    --brain_vae_model_ckpt "${BRAIN_VAE_MODEL_CKPT}" \
    --fMRI_single_trial "${FMRI_SINGLE_TRIAL}" \
    --img_token_ids "${IMG_TOKEN_IDS}" \
    --txt_token_ids "${TXT_TOKEN_IDS}" \
    --fmri_mask_token_path "${FMRI_MASK_TOKEN_PATH}" \
    --brain_vae_codebook_size 128 \
    --brain_vae_token_dim 16 \
    --encoding_loss_weight 1.0 \
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --min_masking_rate 0 \
    --train_batch_size 96 \
    --gradient_accumulation_steps 3 \
    --learning_rate 5e-5 \
    --max_grad_norm 8 \
    --mixed_precision "${MIXED_PRECISION}" \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 300 \
    --use_8bit_adam \
    --allow_tf32 \
    --dataloader_num_workers 4 \
    --checkpoints_total_limit 20 \
    --max_train_steps 50000 \
    --checkpointing_steps 1500 \
    --logging_steps 100 \
    --validation_steps 500 \
    --report_to "wandb" \
    --val_imgs "${VAL_IMGS[@]}" \
    --val_text "${VAL_TEXT[@]}" \
    --val_brain "${VAL_BRAIN[@]}"
