# Mind-Omni

Mind-Omni is a unified brain-vision-language framework built on top of Muddit-style masked diffusion modeling. This repository contains the released training and inference code for perceptual decoding, fMRI tokenization, joint brain-image-text modeling, and stage-2 instruction tuning.

If you are using Codex skills or an agent workflow, start with:

- [`skills/muddit-runbook/SKILL.md`](skills/muddit-runbook/SKILL.md) for the full stage-by-stage runbook
- [`skills/nsd-dataset-usage/SKILL.md`](skills/nsd-dataset-usage/SKILL.md) for dataset layout, path replacement, and launch preparation
- [`skills/ckpt-path-usage/SKILL.md`](skills/ckpt-path-usage/SKILL.md) for replacing local checkpoint roots with the released ModelScope weights

## TODO

- [x] Training code
- [x] Inference code
- [x] Dataset release
- [x] Checkpoint release
- [ ] Evaluation code release

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/ReedOnePeck/Mind-Omni.git
cd Mind-Omni
pip install -r requirements.txt
```

Environment notes:

- Python + PyTorch with `diffusers`, `transformers`, `accelerate`, and `bitsandbytes`
- At least 2 A100 GPUs are required for the full training pipeline
- The released stage-1, stage-1.2, and stage-2 launchers were configured with multi-GPU `accelerate` runs and were originally launched on 4 GPUs
- Before running, set your local dataset root and checkpoint root in the launch or validation files you plan to use

## Dataset

The public dataset release is available on ModelScope:

- Dataset repo: `https://www.modelscope.cn/datasets/LLLLLYYYYYzzz/NSD`

To satisfy ModelScope file-count limits, the dataset is packaged as tar archives rather than millions of loose files.

### Download

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

### Dataset layout after extraction

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

Before running any stage, set the dataset root to your local `NSD_complete` path:

```bash
export NSD_DATA_ROOT=/path/to/NSD_complete

rg -n '/nfs/diskstation/DataStation/public_dataset/NSD_complete|/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete' \
  train_decoder_for_perception \
  train_fMRI_tokenizer_perceptual \
  train_stage1 \
  train_stage1_2 \
  train_stage2_short_VQA \
  Validate_the_models \
  data_processing
```

## Checkpoints

The public checkpoint release is available on ModelScope:

- Checkpoint repo: `https://www.modelscope.cn/models/LLLLLYYYYYzzz/Mind_Omni_V1_ckpt/files`

### Download

Method 1: ModelScope SDK

```bash
pip install -U modelscope
```

```python
from modelscope import snapshot_download

ckpt_dir = snapshot_download("LLLLLYYYYYzzz/Mind_Omni_V1_ckpt")
```

Method 2: Git

```bash
git clone https://www.modelscope.cn/LLLLLYYYYYzzz/Mind_Omni_V1_ckpt.git
```

### Checkpoint layout

The released weights are organized as follows:

```text
Mind-Omni-weights/
`-- Models/
    |-- Muddit/
    |   |-- 1024/
    |   |   `-- mask_token_embedding.pth
    |   |-- 512/
    |   |   `-- transformer/
    |   |       |-- config.json
    |   |       `-- diffusion_pytorch_model.safetensors
    |   |-- scheduler/
    |   |   |-- scheduler.py
    |   |   `-- scheduler_config.json
    |   |-- text_encoder/
    |   |   |-- config.json
    |   |   |-- model.fp16.safetensors
    |   |   `-- model.safetensors
    |   |-- tokenizer/
    |   |   |-- merges.txt
    |   |   |-- special_tokens_map.json
    |   |   |-- tokenizer_config.json
    |   |   `-- vocab.json
    |   `-- vqvae/
    |       |-- config.json
    |       |-- diffusion_pytorch_model.fp16.safetensors
    |       `-- diffusion_pytorch_model.safetensors
    `-- UniBrain/
        |-- fMRI_perceptron/
        |   `-- coarse_and_fine/
        |       `-- checkpoint_epoch_40.pth
        |-- fMRI_tokenizer/
        |   `-- train_with_semantic_perceptual/
        |       `-- token_concat_codebook_size_128_code_dim_16_num_token_64/
        |           `-- checkpoint-14000/
        |               `-- VQ_fMRI/
        |-- train_stage1_2/
        |   |-- checkpoint-24000/
        |   |   |-- config.json
        |   |   `-- pytorch_model.bin
        |   `-- fmri_mask_embedding.pt
        |-- train_stage1_with_encoding/
        |   |-- checkpoint-16500/
        |   |   |-- config.json
        |   |   `-- pytorch_model.bin
        |   `-- fmri_mask_embedding.pt
        `-- train_stage2_shortVQA/
            |-- easy_reasoning/
            |   `-- checkpoint-1800/
            |       |-- config.json
            |       |-- lora_config.json
            |       `-- pytorch_model.bin
            `-- short_detail/
                `-- checkpoint-1200/
                    |-- config.json
                    |-- lora_config.json
                    `-- pytorch_model.bin
```

For later stages, point your shell scripts to these released checkpoints under your local `Mind-Omni-weights` root.

## Training

Run all commands from the repository root after updating the dataset root and checkpoint root in the corresponding `.sh` files.

### Stage 0: Perceptual decoder

```bash
bash train_decoder_for_perception/train_recons_perceptual.sh
```

### Stage 0.5: fMRI tokenizer

```bash
bash train_fMRI_tokenizer_perceptual/train_tokenizer_perceptual.sh
```

### Stage 1: Joint brain-image-text modeling

```bash
bash train_stage1/train_stage1.sh
```

### Stage 1.2: Continued joint training with 64 brain tokens

```bash
bash train_stage1_2/train_stage1_2.sh
```

### Stage 2: Short VQA, detailed captioning, and easy reasoning

```bash
bash train_stage2_short_VQA/train_stage2_shortVQA.sh
```

Before launching stage 2, replace the default warm-start paths in `train_stage2_short_VQA/train_stage2_shortVQA.sh` with the released stage-1.2 checkpoint:

```bash
PRETRAINED_STAGE2_MODEL_ROOT=/path/to/Mind-Omni-weights/Models/UniBrain/train_stage1_2/checkpoint-24000
FMRI_MASK_TOKEN_PATH=/path/to/Mind-Omni-weights/Models/UniBrain/train_stage1_2/fmri_mask_embedding.pt
```

## Inference and Evaluation

Current reference inference entrypoints:

```bash
python train_stage1/validate_stage1.py
python train_stage1_2/validate_stage1_2.py
python train_stage2_short_VQA/validate_stage2_shortVQA.py
```

Notes:

- These validation scripts are reference entrypoints. Set their dataset root, checkpoint root, and output location before running
- Stage-0 and tokenizer inference can be reproduced by loading the released checkpoints under `Mind-Omni-weights/Models/UniBrain/` with the modules in `train_decoder_for_perception/` and `train_fMRI_tokenizer_perceptual/`
- For image reconstruction, you can further feed the decoded initial image together with the generated caption into SDXL or Versatile Diffusion to obtain sharper and more visually appealing images
- Public evaluation code is still being cleaned and will be released later; it remains tracked in the TODO section above

## Citation

```bibtex
@article{lu2026mind,
  title={Mind-Omni: A Unified Multi-Task Framework for Brain-Vision-Language Modeling via Discrete Diffusion},
  author={Lu, Yizhuo and Du, Changde and Shi, Qingyu and Chen, Hang and Peng, Jie and Jiang, Liuyun and Zhao, Shuangchen and He, Huiguang},
  journal={arXiv preprint arXiv:2605.29591},
  year={2026}
}
```
