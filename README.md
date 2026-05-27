# Mind-Omni

Mind-Omni is a brain-to-multimodal generation project built on top of Muddit-style masked diffusion modeling. This repository contains the currently released training and inference code for the full pipeline, including perceptual decoding, fMRI tokenization, joint brain-image-text modeling, and stage-2 VQA-style instruction tuning.

## TODO

- [x] Training code
- [x] Inference code
- [ ] Dataset release (planned within 2 weeks)
- [ ] Evaluation code release (planned within 2 weeks)

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

## Data and Checkpoints

The current codebase uses absolute paths inside the provided shell scripts and validation files. Before running any stage, update those paths to match your local environment.

At a high level, the pipeline expects the following assets:

- fMRI single-trial and multi-trial arrays
- Image token IDs and text token IDs
- CLIP image/text features and text hidden states
- Muddit base model components: transformer config, transformer weights, tokenizer, text encoder, VQ-VAE, and scheduler
- Trained checkpoints from earlier stages when launching later stages

Dataset release is planned within 2 weeks. Until then, the included scripts should be treated as reference implementations for reproducing the pipeline on internally prepared data layouts.

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
