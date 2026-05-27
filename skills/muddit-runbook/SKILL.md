---
name: muddit-runbook
description: Bring up Muddit-132-official on the current server, including dataset locations, stage-wise training commands, tunable arguments, verified checkpoints, loading, and inference entrypoints.
---

# Muddit Runbook

This skill is the current-server runbook for `/data/home/luyizhuo/Python_project/code/Muddit-132-official`.

## 1. Environment

- Activate env: `conda activate muddit`
- Repo root: `/data/home/luyizhuo/Python_project/code/Muddit-132-official`
- Check GPUs first: `nvidia-smi`
- Select GPUs explicitly before launching:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

- Most training launchers already set `GPU_IDS` and call `accelerate launch`. Run them from repo root.

## 2. Path Mapping

Old code and old scripts may still reference `/nfs/...`. On this server use:

- `/nfs/diskstation/DataStation/public_dataset/NSD_complete`
  -> `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete`
- `/nfs/.../Muddit/...`
  -> `/data/home/luyizhuo/Datastation_lyz/Models/Muddit`
- `/nfs/.../UniBrain/...`
  -> `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain`

If a validation or debug script still contains `/nfs/...`, replace it before running.

## 3. Dataset Layout

Main dataset root:

```text
/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete
```

Frequently used subpaths:

- fMRI single-trial: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_single`
- fMRI multi-trial: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi`
- NSD images: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs`
- COCO recaption text: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL`
- image VQ token ids: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/VQVAE_feature_img`
- text token ids: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/caption_ids_COCO_recaption`
- CLIP image features: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy`
- CLIP text features: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_feature_1024/text/text_CLIP_H_feature_1024.npy`
- CLIP text hidden states: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_H_text_max30`
- BQA root: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data`
- short VQA token ids: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA_token_ids`
- detailed caption token ids: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/detail_token_ids`
- easy reasoning token ids: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning_token_ids`
- short question lengths: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA_Q_len.npy`
- detail question lengths: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/detail_Q_len.npy`
- easy reasoning lengths: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning_Q_len.npy`

Stage 2 training should use:

- `short_VQA_token_ids`
- `detail_token_ids`
- `easy_reasoning_token_ids`

Do not switch Stage 2 to `complex_reasoning_token_ids` unless you are intentionally running a different data recipe.

## 4. Base Model Assets

Muddit base model:

- transformer config: `/data/home/luyizhuo/Datastation_lyz/Models/Muddit/512/transformer/config.json`
- transformer weights: `/data/home/luyizhuo/Datastation_lyz/Models/Muddit/512/transformer/diffusion_pytorch_model.safetensors`
- text encoder: `/data/home/luyizhuo/Datastation_lyz/Models/Muddit/text_encoder`
- tokenizer: `/data/home/luyizhuo/Datastation_lyz/Models/Muddit/tokenizer`
- VQ-VAE: `/data/home/luyizhuo/Datastation_lyz/Models/Muddit/vqvae`
- scheduler: `/data/home/luyizhuo/Datastation_lyz/Models/Muddit/scheduler`
- text mask token embedding: `/data/home/luyizhuo/Datastation_lyz/Models/Muddit/1024/mask_token_embedding.pth`

## 5. Stage 0: Perceptual Mapper

Code:

- launcher: `train_decoder_for_perception/train_recons_perceptual.sh`
- trainer: `train_decoder_for_perception/train_recons_perceptual.py`
- model: `train_decoder_for_perception/fMRI_recons_perceptual.py`

Run:

```bash
bash train_decoder_for_perception/train_recons_perceptual.sh
```

Most important knobs:

- `--GPU_ID`
- `--output_dir`
- `--num_train_epochs`
- `--train_batch_size`
- `--val_batch_size`
- `--learning_rate`
- `--checkpointing_epochs`
- `--validation_epochs`
- `--mlp_hidden_dims`

Verified trained checkpoint:

- `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_perceptron/coarse_and_fine/checkpoint_epoch_40.pth`

Minimal load and inference:

```python
import torch
from train_decoder_for_perception.fMRI_recons_perceptual import fMRI_recons_perceptron

ckpt = torch.load(
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_perceptron/coarse_and_fine/checkpoint_epoch_40.pth",
    map_location="cpu",
)

model = fMRI_recons_perceptron(
    input_dim=16127,
    output_dim1=1024,
    output_dim2=29 * 1024,
    hidden_dims=[4096, 4096, 4096, 4096],
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

pred_img_feat, pred_txt_hidden = model(fmri_tensor)
```

This stage predicts CLIP image features and CLIP text hidden features from fMRI. It is used again inside the tokenizer.

## 6. Stage 0.5: fMRI Tokenizer

Code:

- launcher: `train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.sh`
- trainer: `train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.py`
- model: `train_fMRI_tokenizer_perceptual/fMRI_tokenizer_perceptual.py`

Run:

```bash
bash train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.sh
```

Must-keep tokenizer settings:

- `codebook_size = 128`
- `codebook_embed_dim = 16`
- `desired_token_num = 64`
- token aggregation is `concat`, not `mean`

Most important knobs:

- `GPU_IDS`
- `NUM_PROCESSES`
- `MIXED_PRECISION`
- `TRAIN_SUBJECTS`
- `VAL_SUBJECTS`
- `SUBJECT_DATA_RATIO`
- `MASK_RATIO`
- `--train_batch_size`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--checkpointing_steps`
- `--max_train_steps`

Verified trained checkpoint:

- `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI`

Minimal load and inference:

```python
from train_fMRI_tokenizer_perceptual.fMRI_tokenizer_perceptual import VQ_fMRI

brain_vae = VQ_fMRI.from_pretrained(
    "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI"
)
brain_vae.eval()

quantized_tokens, codebook_indices = brain_vae.forward_for_inference(fmri_tensor)
recons_pcc = brain_vae.calculate_pcc(fmri_tensor)
```

Notes:

- `VQ_fMRI.from_pretrained(...)` is the verified loading path.
- This checkpoint already works with the current codebase.

## 7. Stage 1: Joint Image+Text <-> Brain

Tasks used in Stage 1:

- `(image + text) -> brain`
- `brain -> (image + text)`

Code:

- launcher: `train_stage1/train_stage1.sh`
- trainer: `train_stage1/train_mind_omni_stage1.py`
- validation example: `train_stage1/validate_stage1.py`

Run:

```bash
bash train_stage1/train_stage1.sh
```

Most important knobs:

- `GPU_IDS`
- `NUM_PROCESSES`
- `MIXED_PRECISION`
- `OUTPUT_DIR`
- `RESUME_FROM_CHECKPOINT`
- `--train_batch_size`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--lr_scheduler`
- `--lr_warmup_steps`
- `--max_train_steps`
- `--checkpointing_steps`
- `--validation_steps`

Verified trained checkpoint:

- `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_with_encoding/checkpoint-16500`
- mask embedding used by inference: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_with_encoding/fmri_mask_embedding.pt`

Load and inference pattern:

1. Build `Trimodal_SymmetricTransformer2DModel` from `config.json`.
2. Load `pytorch_model.bin`.
3. If keys are prefixed with `_orig_mod.`, strip the prefix first.
4. Create `UnifiedPipeline(...)`.
5. Call the pipeline for the desired task.

Minimal checkpoint load skeleton:

```python
import json
import torch
from MindOmni_src.tri_modal_transformer import Trimodal_SymmetricTransformer2DModel

checkpoint_path = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_with_encoding/checkpoint-16500"
with open(f"{checkpoint_path}/config.json", "r") as f:
    config = json.load(f)

if hasattr(Trimodal_SymmetricTransformer2DModel, "from_config"):
    model = Trimodal_SymmetricTransformer2DModel.from_config(config)
else:
    model = Trimodal_SymmetricTransformer2DModel(**config)

state_dict = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location="cpu")
state_dict = {
    (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
    for k, v in state_dict.items()
}
model.load_state_dict(state_dict, strict=True)
model.eval()
```

The repo’s `train_stage1/validate_stage1.py` is a hardcoded example script. Before running it, replace the checkpoint path, model asset paths, and output path with current-server paths.

## 8. Stage 1-2: Add Single-Modal Brain/Image/Text Tasks

Tasks used in Stage 1-2:

- `(image + text) -> brain`
- `brain -> (image + text)`
- `brain -> image`
- `brain -> text`
- `image -> brain`
- `text -> brain`

Code:

- launcher: `train_stage1_2/train_stage1_2.sh`
- trainer: `train_stage1_2/train_mind_omni_stage1_2.py`
- validation example: `train_stage1_2/validate_stage1_2.py`

Run:

```bash
bash train_stage1_2/train_stage1_2.sh
```

Most important knobs:

- `OUTPUT_DIR`
- `RESUME_FROM_CHECKPOINT`
- `FMRI_MASK_TOKEN_PATH`
- `--train_batch_size`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--max_train_steps`
- `--checkpointing_steps`
- `--validation_steps`
- `--num_of_brain_token`

Recommended checkpoint for the current codebase:

- `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/checkpoint-24000`
- mask embedding: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt`

Important:

- Use `train_stage1_2/checkpoint-24000` for current-code loading and inference.
- Do not use `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage2_mixed/checkpoint-4500` as the primary Stage 1-2 checkpoint for this repo. It is a legacy checkpoint with key mismatch against the current cleaned code.

Loading is the same pattern as Stage 1. `train_stage1_2/validate_stage1_2.py` is the working example entrypoint, but it is also hardcoded and should be edited before use.

## 9. Stage 2: BQA + Cross-Modal Generation

Tasks supported by the current Stage 2 code:

- `(image + text) -> brain`
- `brain -> (image + text)`
- `brain -> image`
- `brain -> text`
- `image -> brain`
- `text -> brain`
- `BQA`

BQA data used by the current code:

- `short_VQA_token_ids`
- `detail_token_ids`
- `easy_reasoning_token_ids`

Code:

- launcher: `train_stage2_short_VQA/train_stage2_shortVQA.sh`
- trainer: `train_stage2_short_VQA/train_stage2_shortVQA.py`
- validation example: `train_stage2_short_VQA/validate_stage2_shortVQA.py`
- LoRA utilities: `train_stage2_short_VQA/lora_checkpoint_utils.py`

Run:

```bash
bash train_stage2_short_VQA/train_stage2_shortVQA.sh
```

Before running Stage 2, set the warm-start source to the current Stage 1-2 checkpoint:

- `PRETRAINED_STAGE2_MODEL_ROOT=/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/checkpoint-24000`
- `FMRI_MASK_TOKEN_PATH=/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt`

Do not keep the old `train_stage2_mixed/checkpoint-4500` as the default Stage 2 warm start.

Most important knobs:

- `SHORT_VQA`
- `DETAILED_CAPTION`
- `EASY_REASONING`
- `OUTPUT_DIR`
- `PRETRAINED_STAGE2_MODEL_ROOT`
- `FMRI_MASK_TOKEN_PATH`
- `--train_batch_size`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--lr_scheduler`
- `--max_train_steps`
- `--checkpointing_steps`
- `--validation_steps`
- `--vqa_loss_weight`
- `--encoding_loss_weight`
- `--use_lora`
- `--lora_r`
- `--lora_alpha`
- `--lora_target_modules`

LoRA status:

- Stage 2 training is LoRA-based.
- Stage 2 inference now auto-detects LoRA checkpoints and auto-attaches adapters before loading weights.
- New Stage 2 checkpoints save `lora_config.json`, so later inference does not need manual LoRA reconstruction.

Verified Stage 2 checkpoints:

- short/detail VQA family: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage3_shortVQA/checkpoint-1200`
- reasoning family: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage3_easy_reason/checkpoint-1800`

Load and inference notes:

- `train_stage2_short_VQA/validate_stage2_shortVQA.py` is the reference entrypoint.
- It already uses `ensure_lora_adapter_for_checkpoint(...)` to attach LoRA before weight loading.
- Edit `checkpoint_path`, `global_step`, output paths, and the task-specific sample section before running.

Checkpoint loading pattern:

```python
import json
import os
import torch
from MindOmni_src_stage2.r_tri_modal_transformer import Trimodal_SymmetricTransformer2DModel
from train_stage2_short_VQA.lora_checkpoint_utils import ensure_lora_adapter_for_checkpoint

checkpoint_path = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage3_shortVQA/checkpoint-1200"

with open(os.path.join(checkpoint_path, "config.json"), "r") as f:
    config = json.load(f)

model = Trimodal_SymmetricTransformer2DModel(**config)
ensure_lora_adapter_for_checkpoint(
    model=model,
    checkpoint_path=os.path.join(checkpoint_path, "pytorch_model.bin"),
    fallback_lora_alpha=16,
    fallback_target_modules=[
        "to_q", "to_k", "to_v", "to_out",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        "proj_mlp", "proj_out", "FFN_proj_in", "FFN_proj_out",
    ],
    fallback_use_dora=True,
)

state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
state_dict = {
    (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
    for k, v in state_dict.items()
}
model.load_state_dict(state_dict, strict=True)
model.eval()
```

We already verified Stage 2 by running actual samples for all supported task families, not just a bare forward pass.

## 10. Fast Parameter Tuning Rules

For quick smoke tests, reduce:

- `max_train_steps`
- `checkpointing_steps`
- `validation_steps`
- `num_train_epochs` for the perceptual mapper

Typical quick-test settings:

- `max_train_steps=2~10`
- `checkpointing_steps=1~4`
- `validation_steps=2~4`
- perceptual `num_train_epochs=2~3`

For VRAM pressure, prefer changing:

- `CUDA_VISIBLE_DEVICES`
- `GPU_IDS`
- `NUM_PROCESSES`
- `train_batch_size`
- `gradient_accumulation_steps`
- `MIXED_PRECISION=bf16`

## 11. Known Good Checkpoint Summary

- Perceptual: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_perceptron/coarse_and_fine/checkpoint_epoch_40.pth`
- Tokenizer: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI`
- Stage 1: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_with_encoding/checkpoint-16500`
- Stage 1-2: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage1_2/checkpoint-24000`
- Stage 2 short/detail: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage3_shortVQA/checkpoint-1200`
- Stage 2 reasoning: `/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/train_stage3_easy_reason/checkpoint-1800`

## 12. Practical Advice

- Prefer editing the provided shell launchers instead of building long one-off commands from scratch.
- The validate scripts are real examples, but they are not generic CLIs. They contain hardcoded checkpoint and output paths and should be edited before use.
- For Stage 2, keep LoRA enabled in training and let the inference loader auto-detect adapters.
- For current-code reproduction, prefer the checkpoints listed in this file over older historical directories with similar names.
