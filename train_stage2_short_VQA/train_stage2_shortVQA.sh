#!/usr/bin/env bash
set -euo pipefail

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline

# chmod +x train_stage2_short_VQA/train_stage2_shortVQA.sh
# ./train_stage2_short_VQA/train_stage2_shortVQA.sh

GPU_IDS="1,2,3,4"
NUM_PROCESSES=4
MAIN_PROCESS_PORT=25000
MIXED_PRECISION="bf16"

PRETRAINED_STAGE2_MODEL_ROOT="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage2_mixed/checkpoint-4500"
TXT_MASK_TOKEN_FILE="/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth"
BRAIN_VAE_MODEL_CKPT="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI"
FMRI_SINGLE_TRIAL="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_single/"
FMRI_MULTI_TRIAL="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/"
IMG_TOKEN_IDS="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/VQVAE_feature_img/"
TXT_TOKEN_IDS="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/caption_ids_COCO_recaption/"
SHORT_VQA="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA_token_ids/"
DETAILED_CAPTION="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/detail_token_ids/"
EASY_REASONING="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning_token_ids/"
Q_LEN_SHORT_VQA_PATH="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA_Q_len.npy"
Q_LEN_CAPTION_PATH="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/detail_Q_len.npy"
Q_LEN_REASONING_PATH="/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning_Q_len.npy"
OUTPUT_DIR="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage2_shortVQA/"
FMRI_MASK_TOKEN_PATH="/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage2_mixed/fmri_mask_embedding.pt"

VAL_IMGS=(
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/46002.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/48617.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/44980.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/32625.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/53052.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/04930.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/06431.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/70335.png"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs/36576.png"
)

VAL_TEXT=(
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/46002.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/48617.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/44980.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/32625.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/53052.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/04930.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/06431.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/70335.txt"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL/36576.txt"
)

VAL_BRAIN=(
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/46002.npy"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/48617.npy"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/44980.npy"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/32625.npy"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/53052.npy"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/04930.npy"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/06431.npy"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/70335.npy"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/36576.npy"
)

VAL_DETAIL_Q=(
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/46002_short_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/48617_short_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/44980_short_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/32625_short_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/53052_short_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/04930_short_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/06431_short_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/70335_short_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/36576_short_q.txt"
)

VAL_REASON_Q=(
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/46002_easy_reason_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/48617_easy_reason_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/44980_easy_reason_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/32625_easy_reason_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/53052_easy_reason_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/04930_easy_reason_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/06431_easy_reason_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/70335_easy_reason_q.txt"
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/sub1_test_data_10/36576_easy_reason_q.txt"
)

VAL_BRAIN_AUX=(
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/38310.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/15939.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/07207.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/07840.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/62302.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/40575.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/45595.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/26598.npy"
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/aux_data/04786.npy"
)

PYTHON_PATH="./" accelerate launch \
    --multi_gpu \
    --gpu_ids "${GPU_IDS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --num_processes "${NUM_PROCESSES}" \
    train_stage2_short_VQA/train_stage2_shortVQA.py \
    --pretrained_stage2_model_root "${PRETRAINED_STAGE2_MODEL_ROOT}" \
    --txt_mask_token_file "${TXT_MASK_TOKEN_FILE}" \
    --brain_vae_model_ckpt "${BRAIN_VAE_MODEL_CKPT}" \
    --fMRI_single_trial "${FMRI_SINGLE_TRIAL}" \
    --fMRI_multi_trial "${FMRI_MULTI_TRIAL}" \
    --img_token_ids "${IMG_TOKEN_IDS}" \
    --txt_token_ids "${TXT_TOKEN_IDS}" \
    --short_vqa "${SHORT_VQA}" \
    --Q_len_short_vqa_path "${Q_LEN_SHORT_VQA_PATH}" \
    --detailed_caption "${DETAILED_CAPTION}" \
    --easy_reasoning "${EASY_REASONING}" \
    --Q_len_caption_path "${Q_LEN_CAPTION_PATH}" \
    --Q_len_reasoning_path "${Q_LEN_REASONING_PATH}" \
    --fmri_mask_token_path "${FMRI_MASK_TOKEN_PATH}" \
    --brain_vae_codebook_size 128 \
    --brain_vae_token_dim 16 \
    --num_of_brain_token 64 \
    --encoding_loss_weight 1.0 \
    --vqa_loss_weight 0.4 \
    --output_dir "${OUTPUT_DIR}" \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules "to_q" "to_k" "to_v" "to_out" "add_q_proj" "add_k_proj" "add_v_proj" "to_add_out" "proj_mlp" "proj_out" "FFN_proj_in" "FFN_proj_out" \
    --min_masking_rate 0 \
    --train_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --learning_rate 6e-5 \
    --max_grad_norm 10 \
    --mixed_precision "${MIXED_PRECISION}" \
    --lr_scheduler cosine \
    --use_8bit_adam \
    --allow_tf32 \
    --dataloader_num_workers 4 \
    --checkpoints_total_limit 200 \
    --max_train_steps 10000 \
    --checkpointing_steps 300 \
    --logging_steps 100 \
    --validation_steps 300 \
    --report_to "wandb" \
    --val_imgs "${VAL_IMGS[@]}" \
    --val_text "${VAL_TEXT[@]}" \
    --val_brain "${VAL_BRAIN[@]}" \
    --val_detail_q "${VAL_DETAIL_Q[@]}" \
    --val_reason_q "${VAL_REASON_Q[@]}" \
    --val_brain_aux "${VAL_BRAIN_AUX[@]}"
