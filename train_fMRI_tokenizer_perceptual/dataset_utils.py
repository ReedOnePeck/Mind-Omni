import sys
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
import torch
from collections import defaultdict
import os
from PIL import Image, ImageDraw, ImageFont
import glob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)


class fMRI_tokenizer_TrainDataset(Dataset):
    def __init__(
        self,
        fMRI_single_root,
        fMRI_multi_root,
        img_feature,
        text_feature,
        text_hidden_root,
        train_subjects: Optional[list[int]] = None,
        subject_data_ratio: float = 1.0,
        seed: int = 20020816,
    ):
        super().__init__()
        self.fMRI_single_root = fMRI_single_root
        self.fMRI_multi_root = fMRI_multi_root
        self.text_hidden_root = text_hidden_root
        self.train_subjects = sorted(train_subjects) if train_subjects is not None else None
        self.subject_data_ratio = subject_data_ratio
        self.seed = seed

        print(">>> Initializing dataset...")

        self.img_feature = torch.from_numpy(img_feature).float()
        self.text_feature = torch.from_numpy(text_feature).float()

        assert self.img_feature.shape[0] == self.text_feature.shape[0], \
            "Image and text features must have the same number of samples."

        print(f"Scanning fMRI files in {fMRI_single_root}...")

        all_fmri_files = sorted(glob.glob(os.path.join(self.fMRI_single_root, "*_trial*.npy")))

        if not all_fmri_files:
            raise FileNotFoundError(f"No fMRI files found in {fMRI_single_root}. Please check the path.")

        file_paths_by_index = defaultdict(list)
        for file_path in all_fmri_files:
            common_idx = int(os.path.basename(file_path).split('_')[0])
            file_paths_by_index[common_idx].append(file_path)

        self.fmri_file_list = self._build_training_file_list(file_paths_by_index)

        self.data_len = len(self.fmri_file_list)
        print(f"Dataset initialized successfully. Found {self.data_len} available fMRI samples.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fmri_file_path = self.fmri_file_list[index]

        # 从文件名中提取 5 位数的公共索引字符串（例如 "00123"）
        # 文件名格式：.../00123_trial1.npy -> 提取 "00123"
        common_idx_str = os.path.basename(fmri_file_path).split('_')[0]

        # 将公共索引字符串转换为整数，用于在特征张量中索引
        common_idx_int = int(common_idx_str)

        fmri_data = torch.from_numpy(np.load(fmri_file_path)).float()

        text_hidden_path = os.path.join(self.text_hidden_root, f"{common_idx_str}.pt")
        text_hidden_dict = torch.load(text_hidden_path, map_location='cpu', weights_only=True)
        input_ids = text_hidden_dict['input_ids']
        attention_mask = text_hidden_dict['attention_mask']
        text_hidden_state = text_hidden_dict.get('hidden_state', text_hidden_dict.get('last_hidden_state'))


        image_clip_feature = self.img_feature[common_idx_int]
        text_clip_feature = self.text_feature[common_idx_int]

        sample_dict = {
            'fmri_data': fmri_data,
            'image_clip_feature': image_clip_feature,
            'text_clip_feature': text_clip_feature,
            'text_input_ids': input_ids,
            'text_attention_mask': attention_mask,
            'text_hidden_state': text_hidden_state,
            'common_index': common_idx_int
        }

        return sample_dict

    def _build_training_file_list(self, file_paths_by_index):
        if self.train_subjects is None:
            return self._apply_subject_ratio(sorted([
                file_path
                for file_list in file_paths_by_index.values()
                for file_path in file_list
            ]))

        subject_to_files = {}
        for subject_id in self.train_subjects:
            subject_index_path = os.path.join(
                self.fMRI_multi_root,
                f"sub{subject_id}_train_img_index_start_from0.npy",
            )
            if not os.path.exists(subject_index_path):
                raise FileNotFoundError(
                    f"Subject index file not found for subject {subject_id}: {subject_index_path}"
                )

            subject_indices = np.load(subject_index_path).astype(int).tolist()
            subject_files = []
            for common_idx in subject_indices:
                subject_files.extend(file_paths_by_index.get(common_idx, []))

            if not subject_files:
                raise ValueError(f"No training files found for subject {subject_id}.")

            subject_to_files[subject_id] = sorted(subject_files)

        return self._apply_subject_ratio(subject_to_files)

    def _apply_subject_ratio(self, fmri_file_map):
        if not (0.0 < self.subject_data_ratio <= 1.0):
            raise ValueError(f"subject_data_ratio must be in (0, 1], got {self.subject_data_ratio}")

        if isinstance(fmri_file_map, list):
            if self.subject_data_ratio >= 1.0:
                return fmri_file_map

            rng = random.Random(self.seed)
            keep_count = max(1, int(round(len(fmri_file_map) * self.subject_data_ratio)))
            sampled_indices = sorted(rng.sample(range(len(fmri_file_map)), keep_count))
            return [fmri_file_map[idx] for idx in sampled_indices]

        if self.subject_data_ratio >= 1.0:
            merged_files = []
            for subject_id in sorted(fmri_file_map):
                merged_files.extend(fmri_file_map[subject_id])
            return sorted(dict.fromkeys(merged_files))

        rng = random.Random(self.seed)
        sampled_files = []
        for subject_id in sorted(fmri_file_map):
            subject_files = fmri_file_map[subject_id]
            keep_count = max(1, int(round(len(subject_files) * self.subject_data_ratio)))
            sampled_indices = sorted(rng.sample(range(len(subject_files)), keep_count))
            sampled_files.extend(subject_files[idx] for idx in sampled_indices)

        return sorted(dict.fromkeys(sampled_files))




class fMRI_tokenizer_ValDataset(Dataset):
    def __init__(
        self,
        fMRI_single_root,
        fMRI_multi_root,
        img_feature,
        text_feature,
        text_hidden_root,
        subjects_to_use: Optional[list[int]] = None,
    ):
        super().__init__()

        self.text_hidden_root = text_hidden_root
        self.img_feature = torch.from_numpy(img_feature).float()
        self.text_feature = torch.from_numpy(text_feature).float()

        print(">>> Initializing validation dataset (Reconstructed version)...")

        self.subjects_to_use = sorted(subjects_to_use) if subjects_to_use is not None else [1, 5]


        base_subject = self.subjects_to_use[0]
        multi_trial_folder_base = os.path.join(fMRI_multi_root, f"test_data_sub{base_subject}")
        index_file_path = os.path.join(multi_trial_folder_base, "test_img_index_start_from0.npy")

        if not os.path.exists(index_file_path):
            raise FileNotFoundError(f"Base index file not found for subject {base_subject} at {index_file_path}.")

        self.common_indices = np.load(index_file_path)
        self.data_len = len(self.common_indices)


        # { subject_id: { "multi": data, "single_1": data, ... }, ... }
        self.all_fmri_data = defaultdict(dict)

        for sub in self.subjects_to_use:
            print(f"Pre-loading all fMRI data for subject {sub}...")

            multi_trial_folder = os.path.join(fMRI_multi_root, f"test_data_sub{sub}")
            multi_fmri_path = os.path.join(multi_trial_folder, f"sub{sub}_test_multi.npy")
            if os.path.exists(multi_fmri_path):
                self.all_fmri_data[sub]["multi"] = np.load(multi_fmri_path)
            else:
                print(f"Warning: Multi-trial data not found for sub {sub}, will be skipped.")
                self.all_fmri_data[sub]["multi"] = None

            single_trial_folder = os.path.join(fMRI_single_root, f"test_data_sub{sub}")
            for trial_num in [1, 2, 3]:
                trial_key = f"single_{trial_num}"
                single_fmri_path = os.path.join(single_trial_folder, f"sub{sub}_test_single_trial{trial_num}.npy")
                if os.path.exists(single_fmri_path):
                    self.all_fmri_data[sub][trial_key] = np.load(single_fmri_path)
                else:
                    print(f"Warning: Single-trial {trial_num} data not found for sub {sub}, will be skipped.")
                    self.all_fmri_data[sub][trial_key] = None
        print(f"Validation dataset initialized successfully. Number of unique stimuli: {self.data_len}.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        """
        index (int): 刺激的索引 (0 to 999)。
        """
        # --- 1. 获取当前刺激对应的公共索引 (在 0-72999 范围内) ---
        common_idx_int = self.common_indices[index]
        common_idx_str = str(common_idx_int).zfill(5)

        # --- 2. 加载共享的多媒体特征 ---
        # 文本隐藏状态是按需加载的
        text_hidden_path = os.path.join(self.text_hidden_root, f"{common_idx_str}.pt")
        text_hidden_dict = torch.load(text_hidden_path, map_location='cpu', weights_only=True)

        image_clip_feature = self.img_feature[common_idx_int]
        text_clip_feature = self.text_feature[common_idx_int]

        # --- 3. 收集与该刺激对应的所有 fMRI 数据 ---
        fmri_data_all_subjects = {}
        for sub in self.subjects_to_use:
            subject_fmri = {}
            # 从预加载的数据中，根据刺激索引 `index` 提取 fMRI 数据
            if self.all_fmri_data[sub]["multi"] is not None:
                subject_fmri["multi"] = torch.from_numpy(self.all_fmri_data[sub]["multi"][index]).float()

            single_trials = []
            for trial_num in [1, 2, 3]:
                trial_key = f"single_{trial_num}"
                if self.all_fmri_data[sub][trial_key] is not None:
                    single_trials.append(
                        torch.from_numpy(self.all_fmri_data[sub][trial_key][index]).float()
                    )

            # 将 3 个 single trials 堆叠成一个张量
            if single_trials:
                subject_fmri["single_stacked"] = torch.stack(single_trials, dim=0)

            fmri_data_all_subjects[f"subject_{sub}"] = subject_fmri

        # --- 4. 组装成最终的字典并返回 ---
        sample_dict = {
            # 共享的多媒体特征
            'image_clip_feature': image_clip_feature,
            'text_clip_feature': text_clip_feature,
            'text_input_ids': text_hidden_dict['input_ids'],
            'text_attention_mask': text_hidden_dict['attention_mask'],
            'text_hidden_state': text_hidden_dict.get('hidden_state', text_hidden_dict.get('last_hidden_state')),

            # 所有 fMRI 数据
            'fmri_data': fmri_data_all_subjects,

            # 元数据
            'stimulus_index': index,  # 刺激索引 (0-999)
            'common_index': common_idx_int,  # 公共索引 (0-72999)
        }

        return sample_dict



if __name__ == '__main__':
    # --- 请将以下路径替换为您的实际路径 ---
    FMR_ROOT = '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_single/'
    IMG_FEAT_PATH = '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy'
    TXT_FEAT_PATH = '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_feature_1024/text/text_CLIP_H_feature_1024.npy'
    TXT_HIDDEN_ROOT = '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_H_text_max30/'
    # -----------------------------------------
    IMG_FEAT = np.load(IMG_FEAT_PATH)
    TXT_FEAT = np.load(TXT_FEAT_PATH)

    # 实例化数据集
    try:
        train_dataset = fMRI_tokenizer_TrainDataset(
            fMRI_single_root=FMR_ROOT,
            img_feature=IMG_FEAT,
            text_feature=TXT_FEAT,
            text_hidden_root=TXT_HIDDEN_ROOT,
        )

        # 使用 DataLoader 来创建数据批次
        # shuffle=True 表示在每个 epoch 开始时都会打乱数据顺序，这对于训练很重要
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

        # 演示如何从 DataLoader 中获取一个批次的数据
        print("\n--- Testing DataLoader ---")
        first_batch = next(iter(train_dataloader))

        # 打印批次中各项数据的形状
        print("Shapes of the first batch:")
        for key, value in first_batch.items():
            print(f"  {key}: {value.shape}")

        # 示例输出：
        # Shapes of the first batch:
        #   fmri_data: torch.Size([16, 16127])
        #   image_clip_feature: torch.Size([16, 1024])
        #   text_clip_feature: torch.Size([16, 1024])
        #   text_input_ids: torch.Size([16, 30])
        #   text_attention_mask: torch.Size([16, 30])
        #   text_hidden_state: torch.Size([16, 30, 1024])
        #   common_index: torch.Size([16])

    except Exception as e:
        print(f"An error occurred: {e}")

    FMR_SINGLE_ROOT = '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_single/'
    FMR_MULTI_ROOT = '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_fMRI_MNI_multi/'
    IMG_FEAT_ARRAY = IMG_FEAT
    TXT_FEAT_ARRAY = TXT_FEAT
    TXT_HIDDEN_ROOT = '/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_H_text_max30/'
    # -----------------------------------------

    try:
        val_dataset = fMRI_tokenizer_ValDataset(
            fMRI_single_root=FMR_SINGLE_ROOT,
            fMRI_multi_root=FMR_MULTI_ROOT,
            img_feature=IMG_FEAT_ARRAY,
            text_feature=TXT_FEAT_ARRAY,
            text_hidden_root=TXT_HIDDEN_ROOT,
        )

        # 验证时，batch_size 通常为 1，这样可以逐个评估每个刺激
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        # 如果 batch_size > 1，你可能需要一个自定义的 collate_fn
        # val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

        print(f"\nTotal unique stimuli in dataset: {len(val_dataset)}")

        # 获取第一个刺激（stimulus_index=0）对应的所有数据
        first_stimulus_data = val_dataset[0]

        print("\n--- Data structure for a single stimulus (index 0) ---")

        # 打印多媒体特征的形状
        print(f"Image CLIP feature shape: {first_stimulus_data['image_clip_feature'].shape}")
        print(f"Text hidden state shape: {first_stimulus_data['text_hidden_state'].shape}")

        # 打印 fMRI 数据的结构和形状
        print("\nfMRI data structure:")
        fmri_data = first_stimulus_data['fmri_data']
        for sub_key, sub_data in fmri_data.items():
            print(f"  - {sub_key}:")
            if "multi" in sub_data:
                print(f"    - multi trial shape: {sub_data['multi'].shape}")
            if "single_stacked" in sub_data:
                print(f"    - single trials stacked shape: {sub_data['single_stacked'].shape}")

        # 示例输出：
        # --- Data structure for a single stimulus (index 0) ---
        # Image CLIP feature shape: torch.Size([1024])
        # Text hidden state shape: torch.Size([30, 1024])
        #
        # fMRI data structure:
        #   - subject_1:
        #     - multi trial shape: torch.Size([16127])
        #     - single trials stacked shape: torch.Size([3, 16127])
        #   - subject_5:
        #     - multi trial shape: torch.Size([16127])
        #     - single trials stacked shape: torch.Size([3, 16127])

    except Exception as e:
        print(f"An error occurred: {e}")
