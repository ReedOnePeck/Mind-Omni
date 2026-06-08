# Mind-Omni

Mind-Omni is a brain-to-multimodal generation project built on top of Muddit-style masked diffusion modeling. This repository contains the currently released training and inference code for the full pipeline, including perceptual decoding, fMRI tokenization, joint brain-image-text modeling, and stage-2 VQA-style instruction tuning.

## TODO

- [x] Training code
- [x] Inference code
- [x] Dataset release (initial ModelScope upload completed)
- [ ] Evaluation code and ckpt release (planned within 2 weeks)

## Overview

The repository is organized as a multi-stage pipeline:

1. `train_decoder_for_perception/`
   Train a perceptual decoder that predicts CLIP-aligned image and text features from fMRI.
2. `train_fMRI_tokenizer_perceptual/`
   Train the fMRI tokenizer that converts brain signals into discrete brain tokens.
3. `train_stage1/`
   Jointly train brain, image, and text alignment for bidirectional brain-to-multimodal generation.
4. `train_stage1_2/`
   Continue stage-1 training with the finalized tokenizer setting and the 64-token brain representation.
5. `train_stage2_short_VQA/`
   Instruction-tune the model for short VQA, detailed captioning, and easy reasoning.
6. `data_processing/`
   Dataset preprocessing utilities for stage-1 and stage-2 training.
7. `Validate_the_models/`
   Internal validation and analysis scripts.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/ReedOnePeck/Mind-Omni.git
cd Mind-Omni
pip install -r requirements.txt
```

Most training scripts assume:

- Python with PyTorch, `diffusers`, `transformers`, `accelerate`, and `bitsandbytes`
- A multi-GPU environment for stage-1 and stage-2 training
- Local access to the required NSD-derived fMRI, image, and text assets

## Dataset

The initial NSD-derived dataset upload has been completed and is available on ModelScope:

- Dataset repo: `https://www.modelscope.cn/datasets/LLLLLYYYYYzzz/NSD`

To satisfy ModelScope file-count limits, the public release is packaged as a small number of tar archives instead of millions of loose files.

### Released archive layout

```text
NSD/
|-- README.md
|-- COCO_IDs.tar
|-- COCO_captions_recapted_Qw2VL.tar
|-- NSD_fMRI_MNI_multi.tar
|-- NSD_fMRI_MNI_single.tar
|-- NSD_features.tar
|-- NSD_imgs.tar
|-- Visual_instruct_tuning_data.tar
|-- nsddata.tar
|-- root_files.tar
`-- short_COCO_caption.tar
```

### Logical dataset layout after extraction

```text
NSD_complete/
|-- COCO_73k_annots_curated.npy
|-- COCO_IDs/
|-- COCO_captions_recapted_Qw2VL/
|-- NSD_fMRI_MNI_multi/
|-- NSD_fMRI_MNI_single/
|-- NSD_features/
|   |-- CLIP_H_text_max30/
|   |-- CLIP_feature_1024/
|   |-- CLIP_feature_Base/
|   |-- VAVAE_feature/
|   |-- VQVAE_continuos_feature/
|   |-- VQVAE_feature_img/
|   `-- caption_ids_COCO_recaption/
|-- NSD_imgs/
|-- Visual_instruct_tuning_data/
|   |-- raw_data/
|   `-- recaptioned_data/
|-- brain2caption_qwen.json
|-- nsddata/
|-- raw_COCO.json
|-- raw_COCO_with_idx.json
|-- short_COCO_caption/
|-- val_stim_multi_trial_data.npy
`-- 数据集说明.txt
```

### Key directories

- `NSD_fMRI_MNI_single/`: single-trial fMRI arrays for each subject
- `NSD_fMRI_MNI_multi/`: multi-trial / test-time fMRI arrays and image index files
- `NSD_features/VQVAE_feature_img/`: image token IDs
- `NSD_features/caption_ids_COCO_recaption/`: text token IDs
- `NSD_features/CLIP_feature_1024/`: CLIP image and text features
- `NSD_features/CLIP_H_text_max30/`: CLIP text hidden states
- `NSD_imgs/`: image files referenced by NSD image indices
- `COCO_captions_recapted_Qw2VL/`: recaptioned text annotations
- `Visual_instruct_tuning_data/recaptioned_data/`: stage-2 short VQA, detailed caption, and easy reasoning data
- `short_COCO_caption/`: short caption data and token IDs
- `COCO_IDs/`: COCO ID mapping files
- `nsddata/`: NSD anatomical / ROI / registration assets

### Download and extraction

Method 1: `git lfs`

```bash
git lfs install
git clone https://www.modelscope.cn/datasets/LLLLLYYYYYzzz/NSD.git
cd NSD

mkdir -p NSD_complete
for f in *.tar; do
  tar -xf "$f" -C NSD_complete
done
```

Method 2: ModelScope CLI

```bash
pip install -U modelscope
modelscope download --dataset LLLLLYYYYYzzz/NSD --local_dir ./NSD-modelscope
cd NSD-modelscope

mkdir -p NSD_complete
for f in *.tar; do
  tar -xf "$f" -C NSD_complete
done
```

After extraction, set your dataset root:

```bash
export NSD_DATA_ROOT=/path/to/NSD_complete
```

### Using the dataset with this repository

Many training and validation scripts still contain hardcoded dataset paths from the internal environment. Before running any stage, locate and replace them with your own `NSD_DATA_ROOT`.

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

For an agent-oriented runbook that explains the dataset directories and the path-replacement workflow, see `skills/nsd-dataset-usage/SKILL.md`.

## Data and Checkpoints

The current codebase uses absolute paths inside the provided shell scripts and validation files. Before running any stage, update those paths to match your local environment.

At a high level, the pipeline expects the following assets:

- fMRI single-trial and multi-trial arrays
- Image token IDs and text token IDs
- CLIP image/text features and text hidden states
- Muddit base model components: transformer config, transformer weights, tokenizer, text encoder, VQ-VAE, and scheduler
- Trained checkpoints from earlier stages when launching later stages

The released dataset packaging is described above. The included scripts should still be treated as reference implementations because many of them require local path edits before use.

## Repository Layout

```text
Mind-Omni/
|-- MindOmni_src/                       # Stage-1 transformer and pipeline
|-- MindOmni_src_stage2/                # Stage-2 transformer and pipeline
|-- MindOmni_utils/                     # Shared training and scheduler utilities
|-- data_processing/                    # Dataset preparation scripts
|-- train_decoder_for_perception/       # Stage 0
|-- train_fMRI_tokenizer_perceptual/    # Stage 0.5
|-- train_stage1/                       # Stage 1
|-- train_stage1_2/                     # Stage 1.2
|-- train_stage2_short_VQA/             # Stage 2
`-- Validate_the_models/                # Validation and analysis scripts
```

## Training

Run all commands from the repository root.

### Stage 0: Perceptual Decoder

Train the perceptual decoder:

```bash
bash train_decoder_for_perception/train_recons_perceptual.sh
```

Main entrypoints:

- Launcher: `train_decoder_for_perception/train_recons_perceptual.sh`
- Trainer: `train_decoder_for_perception/train_recons_perceptual.py`
- Model: `train_decoder_for_perception/fMRI_recons_perceptual.py`

### Stage 0.5: fMRI Tokenizer

Train the tokenizer that maps fMRI to discrete brain tokens:

```bash
bash train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.sh
```

Main entrypoints:

- Launcher: `train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.sh`
- Trainer: `train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.py`
- Model: `train_fMRI_tokenizer_perceptual/fMRI_tokenizer_perceptual.py`

### Stage 1: Joint Brain-Image-Text Modeling

Train the first joint multimodal stage:

```bash
bash train_stage1/train_stage1.sh
```

Main entrypoints:

- Launcher: `train_stage1/train_stage1.sh`
- Trainer: `train_stage1/train_mind_omni_stage1.py`
- Validation / inference example: `train_stage1/validate_stage1.py`

### Stage 1.2: Continued Joint Training

Continue training with the 64-token brain setup:

```bash
bash train_stage1_2/train_stage1_2.sh
```

Main entrypoints:

- Launcher: `train_stage1_2/train_stage1_2.sh`
- Trainer: `train_stage1_2/train_mind_omni_stage1_2.py`
- Validation / inference example: `train_stage1_2/validate_stage1_2.py`

### Stage 2: Short VQA and Instruction Tuning

Train the stage-2 model with short VQA, detailed captioning, and easy reasoning data:

```bash
bash train_stage2_short_VQA/train_stage2_shortVQA.sh
```

Main entrypoints:

- Launcher: `train_stage2_short_VQA/train_stage2_shortVQA.sh`
- Trainer: `train_stage2_short_VQA/train_stage2_shortVQA.py`
- Validation / inference example: `train_stage2_short_VQA/validate_stage2_shortVQA.py`

## Inference

### Stage 0: Perceptual Decoder Inference

Minimal usage:

```python
import torch
from train_decoder_for_perception.fMRI_recons_perceptual import fMRI_recons_perceptron

ckpt = torch.load("path/to/checkpoint_epoch_xx.pth", map_location="cpu")
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

### Stage 0.5: fMRI Tokenizer Inference

Minimal usage:

```python
from train_fMRI_tokenizer_perceptual.fMRI_tokenizer_perceptual import VQ_fMRI

brain_vae = VQ_fMRI.from_pretrained("path/to/VQ_fMRI")
brain_vae.eval()

quantized_tokens, codebook_indices = brain_vae.forward_for_inference(fmri_tensor)
recons_pcc = brain_vae.calculate_pcc(fmri_tensor)
```

### Stage 1: Brain-to-Image and Brain-to-Text Inference

The repository already includes an end-to-end inference example:

```bash
python train_stage1/validate_stage1.py
```

This script shows how to:

- Load the tokenizer, text encoder, VQ-VAE, scheduler, and stage-1 transformer
- Build `UnifiedPipeline`
- Generate images and texts from fMRI inputs

### Stage 1.2: Batch Inference on Test Splits

Use the provided validation script:

```bash
python train_stage1_2/validate_stage1_2.py
```

This script performs batched multimodal decoding from brain signals and saves generated images and texts.

### Stage 2: Short VQA / Caption / Reasoning Inference

Use the provided validation script:

```bash
python train_stage2_short_VQA/validate_stage2_shortVQA.py
```

This script demonstrates:

- Loading a stage-2 checkpoint
- Restoring LoRA adapters when needed
- Building the stage-2 `UnifiedPipeline`
- Running brain-conditioned generation for VQA-style outputs

## Data Processing

Preprocessing utilities are included under `data_processing/`.

- `data_processing/stage1_feature_prep/`
  Scripts for image/text tokenization, feature extraction, and recaption preparation
- `data_processing/stage2_dataset_prep/`
  Scripts for matching, cleaning, recaptioning, and tokenizing stage-2 instruction data

These scripts reflect the internal data preparation workflow used by the released training code.

## Notes

- Many scripts currently contain machine-specific absolute paths and should be edited before use.
- Later stages depend on checkpoints produced by earlier stages.
- Validation and analysis scripts are available in `Validate_the_models/`, while a cleaned public evaluation release is planned within 2 weeks.

## Citation

If you find this repository useful, please cite the corresponding project or paper once the public release information is available.
