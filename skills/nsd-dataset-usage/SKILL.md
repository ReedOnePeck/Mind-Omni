---
name: nsd-dataset-usage
description: Use when running Mind-Omni with a local NSD_complete dataset copy. Inspect the local dataset root, explain what each dataset directory is for, and replace hardcoded dataset paths in scripts before training or inference.

---

# NSD Dataset Usage

This skill is for `/data/home/luyizhuo/Python_project/code/Muddit-132-official` when code needs to run against a local `NSD_complete` dataset.

## 1. Pick the dataset root first

- Prefer a user-provided dataset root.
- Otherwise search likely locations and stop only after finding a directory that contains the expected top-level folders.

```bash
find "$PWD" "$HOME" /data -maxdepth 4 -type d -name NSD_complete 2>/dev/null | sed -n '1,40p'
```

The chosen root should contain at least:

- `NSD_fMRI_MNI_single`
- `NSD_fMRI_MNI_multi`
- `NSD_features`
- `NSD_imgs`

If the dataset came from the released ModelScope repo, extract the tar archives first:

```bash
mkdir -p NSD_complete
for f in *.tar; do
  tar -xf "$f" -C NSD_complete
done
```

## 2. Understand the directory mapping

- `NSD_fMRI_MNI_single/`: single-trial fMRI arrays
- `NSD_fMRI_MNI_multi/`: multi-trial / averaged fMRI arrays and image index files
- `NSD_features/VQVAE_feature_img/`: image token IDs
- `NSD_features/caption_ids_COCO_recaption/`: text token IDs
- `NSD_features/CLIP_feature_1024/`: CLIP image and text features
- `NSD_features/CLIP_H_text_max30/`: CLIP text hidden states
- `NSD_imgs/`: image files
- `COCO_captions_recapted_Qw2VL/`: recaptioned text annotations
- `Visual_instruct_tuning_data/recaptioned_data/short_VQA_token_ids/`: stage-2 short VQA token IDs
- `Visual_instruct_tuning_data/recaptioned_data/detail_token_ids/`: stage-2 detailed caption token IDs
- `Visual_instruct_tuning_data/recaptioned_data/easy_reasoning_token_ids/`: stage-2 easy reasoning token IDs
- `Visual_instruct_tuning_data/recaptioned_data/*_Q_len.npy`: question length arrays used by stage 2
- `short_COCO_caption/`: short-caption data and token IDs
- `COCO_IDs/`: COCO ID mapping files
- `nsddata/`: ROI, registration, and anatomical assets

## 3. Inspect the actual local structure

Before changing any code, inspect the current dataset tree and verify the expected paths exist.

```bash
export NSD_DATA_ROOT=/path/to/NSD_complete

find "$NSD_DATA_ROOT" -maxdepth 2 -mindepth 1 | sed -n '1,200p'
du -sh "$NSD_DATA_ROOT"/*
```

If key directories are missing, stop and report the mismatch instead of guessing.

## 4. Replace hardcoded dataset roots before running

Search for both path families that appear in this repo:

```bash
rg -n '/nfs/diskstation/DataStation/public_dataset/NSD_complete|/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete' .
```

Then replace the dataset path in the files you are actually going to run:

- For shell launchers, edit the variables near the top of the `.sh` file.
- For Python training scripts, edit the `default=` values in `argparse`.
- For validation scripts, edit the local constants and sample paths.
- Use `apply_patch` for repo edits.

Example mapping:

- old root: `/nfs/diskstation/DataStation/public_dataset/NSD_complete`
- old root: `/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete`
- new root: `$NSD_DATA_ROOT`

## 5. Verify before launch

After replacing paths, check that no stale dataset roots remain in the files you plan to execute.

```bash
rg -n '/nfs/diskstation/DataStation/public_dataset/NSD_complete|/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete' \
  train_decoder_for_perception \
  train_fMRI_tokenizer_perceptual \
  train_stage1 \
  train_stage1_2 \
  train_stage2_short_VQA \
  Validate_the_models \
  data_processing
```

Also verify a few expected files exist:

```bash
test -f "$NSD_DATA_ROOT/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy"
test -f "$NSD_DATA_ROOT/NSD_features/CLIP_feature_1024/text/text_CLIP_H_feature_1024.npy"
test -d "$NSD_DATA_ROOT/Visual_instruct_tuning_data/recaptioned_data/short_VQA_token_ids"
test -d "$NSD_DATA_ROOT/NSD_imgs"
```

If dataset paths are fixed but model checkpoint roots are still machine-specific, surface that separately instead of silently changing unrelated paths.
