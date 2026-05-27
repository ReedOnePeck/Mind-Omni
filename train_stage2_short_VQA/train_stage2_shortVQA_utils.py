import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random


class Stage2_TrainDataset(Dataset):
    def __init__(self, fMRI_root: str, image_token_root: str, text_token_root: str, fMRI_single_trial_root: str,
                 short_vqa: str = None, Q_len_short_vqa_path: str = None,
                 detailed_caption: str = None, Q_len_caption_path: str = None,
                 easy_reasoning: str = None, Q_len_reasoning_path: str = None):
        super().__init__()
        self.fMRI_root = fMRI_root
        self.image_token_root = image_token_root
        self.text_token_root = text_token_root
        self.fMRI_single_trial_root = fMRI_single_trial_root

        print(">>> 正在预处理Q_len数据为字典...")
        self.vqa_sources = []
        self.vqa_source_counts = {}
        self._register_vqa_source("short_vqa", short_vqa, Q_len_short_vqa_path)
        self._register_vqa_source("detailed_caption", detailed_caption, Q_len_caption_path)
        self._register_vqa_source("easy_reasoning", easy_reasoning, Q_len_reasoning_path)

        if not self.vqa_sources:
            raise ValueError("至少需要提供一个 VQA 数据源。")

        print(">>> 正在初始化第三阶段三模态数据集 (Multi-trial)...")

        self.gen_samples = []
        self.vqa_samples = []

        # 在初始化时直接调用加载函数
        self._load_gen()
        self._load_vqa()

        self.len_gen = len(self.gen_samples)
        self.len_vqa = len(self.vqa_samples)

        print(f">>> 加载完成! 生成任务样本数: {len(self.gen_samples)}, VQA任务样本数: {len(self.vqa_samples)}")
        for source_name, source_count in self.vqa_source_counts.items():
            print(f"    - VQA子类型 {source_name}: {source_count}")

    def _register_vqa_source(self, source_name, root_path, q_len_path):
        if root_path is None or q_len_path is None:
            return
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"VQA 数据目录不存在: {root_path}")
        if not os.path.isfile(q_len_path):
            raise FileNotFoundError(f"VQA Q_len 文件不存在: {q_len_path}")

        files = sorted(glob.glob(os.path.join(root_path, '*.npy')))
        q_len_values = np.load(q_len_path)
        q_len_map = {
            os.path.splitext(os.path.basename(f))[0]: q_len
            for f, q_len in zip(files, q_len_values)
        }
        self.vqa_sources.append({
            "name": source_name,
            "root": root_path,
            "q_len_map": q_len_map,
        })
        self.vqa_source_counts[source_name] = 0

    def _append_vqa_samples_for_base_id(self, fmri_data_list, image_token_path, base_id):
        for source in self.vqa_sources:
            text_path = os.path.join(source["root"], f"{base_id}.npy")
            if not os.path.exists(text_path):
                continue

            q_len = source["q_len_map"].get(base_id)
            if q_len is None:
                continue

            paired_data = [fmri_data_list, image_token_path, text_path, q_len]
            self.vqa_samples.append(paired_data)
            self.vqa_source_counts[source["name"]] += 1

    def _find_single_trials(self, base_id):
        trial_files = []
        for i in range(1, 4):
            trial_path = os.path.join(self.fMRI_single_trial_root, f"{base_id}_trial{i}.npy")
            if os.path.exists(trial_path):
                trial_files.append(trial_path)
        return trial_files


    def _load_gen(self):
        """
        加载用于生成任务的数据对。
        以 fMRI_root 为基准，匹配 image_token, text_token 和 single_trial_fMRI。
        """
        print(">>> 正在加载 'generation' 任务数据...")
        # 扫描 fMRI_root 目录下的所有 npy 文件
        fMRI_files = sorted(glob.glob(os.path.join(self.fMRI_root, '*.npy')))

        for fMRI_path in tqdm(fMRI_files, desc="处理 gen 数据"):
            # 从fMRI文件名中解析出核心ID，例如 "00123"
            base_name = os.path.basename(fMRI_path)
            base_id = os.path.splitext(base_name)[0]

            # 构建其他文件的路径
            image_token_path = os.path.join(self.image_token_root, f"{base_id}.npy")
            text_token_path = os.path.join(self.text_token_root, f"{base_id}.npy")

            # 查找所有对应的single trial文件
            single_trial_paths = self._find_single_trials(base_id)

            # 检查所有配对文件是否存在
            if os.path.exists(image_token_path) and os.path.exists(text_token_path) and single_trial_paths:
                # 组织fMRI文件路径列表
                fmri_data_list = [fMRI_path] + single_trial_paths

                # 按照指定格式组合数据
                paired_data = [fmri_data_list, image_token_path, text_token_path]
                self.gen_samples.append(paired_data)

    def _load_vqa(self):
        """
        加载用于VQA任务的数据对。
        以 fMRI_root 为基准，收集 short/detail/reasoning 等所有已注册的 VQA 子类型。
        """
        print(">>> 正在加载 'VQA' 任务数据 (以fMRI为基准)...")
        fMRI_files = sorted(glob.glob(os.path.join(self.fMRI_root, '*.npy')))

        for fMRI_path in tqdm(fMRI_files, desc="处理 VQA 数据"):
            base_name = os.path.basename(fMRI_path)
            base_id = os.path.splitext(base_name)[0]

            # VQA 任务都需要 image_token 和 single_trials
            image_token_path = os.path.join(self.image_token_root, f"{base_id}.npy")
            single_trial_paths = self._find_single_trials(base_id)

            # 如果基础的 image_token 或 single_trials 不存在，则跳过此fMRI样本
            if not (os.path.exists(image_token_path) and single_trial_paths):
                continue

            fmri_data_list = [fMRI_path] + single_trial_paths


            self._append_vqa_samples_for_base_id(fmri_data_list, image_token_path, base_id)

    def __len__(self):
        # 假设我们想让一个epoch覆盖所有类型的数据
        return max(len(self.gen_samples),  len(self.vqa_samples))


    def __getitem__(self, idx):
        # --- 1. 获取 Generation 任务样本 ---
        try:
            # 尝试用传入的索引直接获取样本
            gen_sample = self.gen_samples[idx]
        except IndexError:
            # 如果索引超出范围（因为 gen_samples 较短），则随机选择一个
            random_idx = random.randint(0, self.len_gen - 1)
            gen_sample = self.gen_samples[random_idx]

        # 从样本中解包路径
        gen_fmri_list, gen_image_token_path, gen_text_token_path = gen_sample
        # 随机选择一个 fMRI trial
        gen_fmri_path = random.choice(gen_fmri_list)

        gen_fmri = torch.from_numpy(np.load(gen_fmri_path)).float()
        gen_img_ids = torch.from_numpy(np.load(gen_image_token_path)).long()
        gen_txt_ids = torch.from_numpy(np.load(gen_text_token_path)).long()
        gen_micro_conds = torch.tensor([512, 512, 0, 0, 6], dtype=torch.long)


        # --- 2. 获取 VQA 任务样本 ---
        try:
            # 尝试用传入的索引直接获取样本
            vqa_sample = self.vqa_samples[idx]
        except IndexError:
            # 如果索引超出范围（因为 vqa_samples 较短），则随机选择一个
            random_idx = random.randint(0, self.len_vqa - 1)
            vqa_sample = self.vqa_samples[random_idx]

        # 从样本中解包路径和元数据
        vqa_fmri_list, vqa_image_token_path, vqa_text_token_path, vqa_q_len = vqa_sample
        # 随机选择一个 fMRI trial
        vqa_fmri_path = random.choice(vqa_fmri_list)

        vqa_fmri = torch.from_numpy(np.load(vqa_fmri_path)).float()
        vqa_img_ids = torch.from_numpy(np.load(vqa_image_token_path)).long()
        vqa_txt_ids = torch.from_numpy(np.load(vqa_text_token_path)).long()
        vqa_micro_conds = torch.tensor([512, 512, 0, 0, 6], dtype=torch.long)



        # --- 3. 组合成字典返回 ---

        # 为了健壮性，增加一个try-except块处理文件加载失败的情况
        try:
            ret = {
                # Generation 任务相关数据
                "gen_fmri": gen_fmri,
                "gen_image_token": gen_img_ids,
                "gen_text_token": gen_txt_ids,
                "gen_micro_conds": gen_micro_conds,

                # VQA 任务相关数据
                "vqa_fmri": vqa_fmri,
                "vqa_image_token": vqa_img_ids,
                "vqa_text_token": vqa_txt_ids,
                "vqa_micro_conds": vqa_micro_conds,
                "vqa_question_len": torch.LongTensor([vqa_q_len]),
            }
            return ret
        except Exception as e:
            # 如果在获取路径时发生任何预料之外的错误，打印信息并尝试获取下一个样本
            print(f"在 __getitem__ 中处理索引 {idx} 时发生错误: {e}")
            # 简单地递归调用下一个索引，避免因为单个样本问题导致训练中断
            next_idx = (idx + 1) % self.__len__()
            return self.__getitem__(next_idx)


def collate_fn(samples):
    """
    一个为 Stage2_TrainDataset 定制的 collate_fn。

    它接收一个字典列表，并将每个键对应的值堆叠成一个批次。

    Args:
        samples (list): 一个字典的列表，每个字典由 Stage2_TrainDataset.__getitem__ 返回。

    Returns:
        dict: 一个包含批次化张量的字典。
    """
    # 初始化用于收集批次数据的列表
    gen_fmri_list = []
    gen_image_token_list = []
    gen_text_token_list = []
    gen_micro_conds_list = []

    vqa_fmri_list = []
    vqa_image_token_list = []
    vqa_text_token_list = []
    vqa_micro_conds_list = []
    vqa_question_len_list = []

    # 遍历批次中的每一个样本（字典）
    for sample in samples:
        # 收集 Generation 任务的数据
        gen_fmri_list.append(sample["gen_fmri"])
        gen_image_token_list.append(sample["gen_image_token"])
        gen_text_token_list.append(sample["gen_text_token"])
        gen_micro_conds_list.append(sample["gen_micro_conds"])

        # 收集 VQA 任务的数据
        vqa_fmri_list.append(sample["vqa_fmri"])
        vqa_image_token_list.append(sample["vqa_image_token"])
        vqa_text_token_list.append(sample["vqa_text_token"])
        vqa_micro_conds_list.append(sample["vqa_micro_conds"])
        vqa_question_len_list.append(sample["vqa_question_len"])

    # --- 使用 torch.stack 或 torch.cat 将列表堆叠成批次张量 ---

    # torch.stack 会增加一个新的维度（通常是第0维）作为批次维度
    gen_fmri = torch.stack(gen_fmri_list, dim=0)
    gen_image_token = torch.stack(gen_image_token_list, dim=0)
    gen_text_token = torch.stack(gen_text_token_list, dim=0)
    gen_micro_conds = torch.stack(gen_micro_conds_list, dim=0)

    vqa_fmri = torch.stack(vqa_fmri_list, dim=0)
    vqa_image_token = torch.stack(vqa_image_token_list, dim=0)
    vqa_text_token = torch.stack(vqa_text_token_list, dim=0)
    vqa_micro_conds = torch.stack(vqa_micro_conds_list, dim=0)

    # 对于 question_len，它已经是 [1] 形状的张量，我们想把它合并成 [B] 形状的一维张量
    # 使用 torch.cat 更合适
    vqa_question_len = torch.cat(vqa_question_len_list, dim=0)

    # --- 将所有批次化的张量组合成最终的返回字典 ---
    batch = {
        # Generation 任务相关数据
        "gen_fmri": gen_fmri,
        "gen_image_token": gen_image_token,
        "gen_text_token": gen_text_token,
        "gen_micro_conds": gen_micro_conds,

        # VQA 任务相关数据
        "vqa_fmri": vqa_fmri,
        "vqa_image_token": vqa_image_token,
        "vqa_text_token": vqa_text_token,
        "vqa_micro_conds": vqa_micro_conds,
        "vqa_question_len": vqa_question_len,
    }

    return batch

