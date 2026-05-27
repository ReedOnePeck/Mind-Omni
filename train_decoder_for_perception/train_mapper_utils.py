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


class fMRI_perceptron_TrainDataset(Dataset):
    def __init__(self, fMRI_single_root, img_feature, text_feature):
        super().__init__()
        self.fMRI_single_root = fMRI_single_root

        print(">>> Initializing dataset...")

        self.img_feature = torch.from_numpy(img_feature).float()
        self.text_feature = torch.from_numpy(text_feature).float()

        assert self.img_feature.shape[0] == self.text_feature.shape[0], \
            "Image and text features must have the same number of samples."

        print(f"Scanning fMRI files in {fMRI_single_root}...")

        self.fmri_file_list = sorted(glob.glob(os.path.join(self.fMRI_single_root, "*_trial*.npy")))

        if not self.fmri_file_list:
            raise FileNotFoundError(f"No fMRI files found in {fMRI_single_root}. Please check the path.")

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

        image_clip_feature = self.img_feature[common_idx_int]
        text_clip_feature = self.text_feature[common_idx_int]

        sample_dict = {
            'fmri_data': fmri_data,
            'image_clip_feature': image_clip_feature,
            'text_clip_feature': text_clip_feature,
            'common_index': common_idx_int
        }

        return sample_dict




class fMRI_perceptron_ValDataset(Dataset):
    def __init__(self, fMRI_single_root, fMRI_multi_root, img_feature, text_feature):
        super().__init__()

        self.img_feature = torch.from_numpy(img_feature).float()
        self.text_feature = torch.from_numpy(text_feature).float()

        print(">>> Initializing validation dataset (Reconstructed version)...")

        self.subjects_to_use = [1, 5]


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

            # 所有 fMRI 数据
            'fmri_data': fmri_data_all_subjects,

            # 元数据
            'stimulus_index': index,  # 刺激索引 (0-999)
            'common_index': common_idx_int,  # 公共索引 (0-72999)
        }

        return sample_dict



