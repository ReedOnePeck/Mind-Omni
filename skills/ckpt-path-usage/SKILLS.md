---
name: ckpt-path-usage
description: Use when running Mind-Omni with the released ModelScope checkpoints. Inspect the local checkpoint tree, map Muddit and UniBrain roots, and replace internal checkpoint paths in launch or validation scripts before training or inference.
---

# Checkpoint Path Usage

This skill is for `/data/home/luyizhuo/Python_project/code/Muddit-132-official` when code needs to run against a local copy of the released `Mind_Omni_V1_ckpt` weights.

## 1. Pick the checkpoint root first

- Prefer a user-provided checkpoint root.
- Otherwise search likely locations and stop only after finding a directory that contains both `Models/Muddit` and `Models/UniBrain`.

```bash
find "$PWD" "$HOME" /data -maxdepth 5 -type d \( -name Mind-Omni-weights -o -name Mind_Omni_V1_ckpt \) 2>/dev/null | sed -n '1,40p'
```

If the repo was cloned directly from ModelScope, the effective checkpoint root is usually:

- `/path/to/Mind_Omni_V1_ckpt/Mind-Omni-weights`

Export it before editing files:

```bash
export MIND_OMNI_CKPT_ROOT=/path/to/Mind-Omni-weights
```

## 2. Confirm the expected structure

Verify the main released assets exist:

```bash
find "$MIND_OMNI_CKPT_ROOT/Models" -maxdepth 4 | sed -n '1,200p'
```

The root should contain at least:

- `Models/Muddit/512/transformer/config.json`
- `Models/Muddit/512/transformer/diffusion_pytorch_model.safetensors`
- `Models/Muddit/text_encoder/`
- `Models/Muddit/tokenizer/`
- `Models/Muddit/vqvae/`
- `Models/Muddit/scheduler/`
- `Models/Muddit/1024/mask_token_embedding.pth`
- `Models/UniBrain/fMRI_perceptron/coarse_and_fine/checkpoint_epoch_40.pth`
- `Models/UniBrain/fMRI_tokenizer/.../checkpoint-14000/VQ_fMRI`
- `Models/UniBrain/train_stage1_with_encoding/checkpoint-16500/`
- `Models/UniBrain/train_stage1_2/checkpoint-24000/`
- `Models/UniBrain/train_stage2_shortVQA/short_detail/checkpoint-1200/`

If key assets are missing, stop and report the mismatch instead of guessing.

## 3. Search for checkpoint paths in the files you will run

Search for both original internal roots and old local roots:

```bash
rg -n '/nfs/.*/Muddit|/nfs/.*/UniBrain|/data/home/luyizhuo/Datastation_lyz/Models/Muddit|/data/home/luyizhuo/Datastation_lyz/Models/UniBrain' \
  train_decoder_for_perception \
  train_fMRI_tokenizer_perceptual \
  train_stage1 \
  train_stage1_2 \
  train_stage2_short_VQA \
  Validate_the_models \
  data_processing
```

Then update only the files you are actually going to execute:

- For shell launchers, edit the variables near the top of the `.sh` file.
- For Python scripts, edit `argparse default=` values, constants, and sample checkpoint settings.
- Use `apply_patch` for repo edits.

## 4. Map released checkpoints into the repo

Use these path families:

- Muddit root: `$MIND_OMNI_CKPT_ROOT/Models/Muddit`
- UniBrain root: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain`

Common replacements:

- transformer config: `$MIND_OMNI_CKPT_ROOT/Models/Muddit/512/transformer/config.json`
- transformer weights: `$MIND_OMNI_CKPT_ROOT/Models/Muddit/512/transformer/diffusion_pytorch_model.safetensors`
- text encoder: `$MIND_OMNI_CKPT_ROOT/Models/Muddit/text_encoder`
- tokenizer: `$MIND_OMNI_CKPT_ROOT/Models/Muddit/tokenizer`
- VQ-VAE: `$MIND_OMNI_CKPT_ROOT/Models/Muddit/vqvae`
- scheduler: `$MIND_OMNI_CKPT_ROOT/Models/Muddit/scheduler`
- text mask token: `$MIND_OMNI_CKPT_ROOT/Models/Muddit/1024/mask_token_embedding.pth`
- perceptual decoder: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain/fMRI_perceptron/coarse_and_fine/checkpoint_epoch_40.pth`
- fMRI tokenizer: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain/fMRI_tokenizer/train_with_semantic_perceptual/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-14000/VQ_fMRI`
- stage 1: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain/train_stage1_with_encoding/checkpoint-16500`
- stage 1 mask: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain/train_stage1_with_encoding/fmri_mask_embedding.pt`
- stage 1.2: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain/train_stage1_2/checkpoint-24000`
- stage 1.2 mask: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt`
- stage 2 short/detail: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain/train_stage2_shortVQA/short_detail/checkpoint-1200`
- stage 2 reasoning: `$MIND_OMNI_CKPT_ROOT/Models/UniBrain/train_stage2_shortVQA/easy_reasoning/checkpoint-1800`

For stage 2 training, warm-start from the released stage-1.2 checkpoint unless you intentionally want a different source.

## 5. Verify before launch

Check that no stale checkpoint roots remain in the files you plan to run:

```bash
rg -n '/nfs/.*/Muddit|/nfs/.*/UniBrain|/data/home/luyizhuo/Datastation_lyz/Models/Muddit|/data/home/luyizhuo/Datastation_lyz/Models/UniBrain' \
  train_decoder_for_perception \
  train_fMRI_tokenizer_perceptual \
  train_stage1 \
  train_stage1_2 \
  train_stage2_short_VQA \
  Validate_the_models
```

Also verify a few expected files exist:

```bash
test -f "$MIND_OMNI_CKPT_ROOT/Models/Muddit/512/transformer/config.json"
test -f "$MIND_OMNI_CKPT_ROOT/Models/Muddit/512/transformer/diffusion_pytorch_model.safetensors"
test -d "$MIND_OMNI_CKPT_ROOT/Models/UniBrain/train_stage1_2/checkpoint-24000"
test -d "$MIND_OMNI_CKPT_ROOT/Models/UniBrain/train_stage2_shortVQA/short_detail/checkpoint-1200"
```
