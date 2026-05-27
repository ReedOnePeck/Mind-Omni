from bert_score import score
import json
import os
import numpy as np

# --- 1. 配置与函数定义 (基本不变) ---

# Ground Truth 文件路径
file1_path = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning_groundtruth.json"

# 结果文件所在的目录
results_dir = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage3/format_for_evaluate/"

# 定义要处理的被试列表
subjects = ["sub1", "sub2", "sub5", "sub7"]


def extract_sentences_from_json(file_path, is_list_value=False):
    """
    从JSON文件提取句子并按数字键排序。
    """
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return []

    sorted_keys = sorted(data.keys(), key=lambda x: int(x))
    for key in sorted_keys:
        value = data[key]
        if is_list_value:
            sentences.append(value[0])
        else:
            sentences.append(value)
    return sentences


def load_captions_from_results(file_path):
    """从结果json文件中加载字幕"""
    captions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"警告：结果文件 {file_path} 未找到，将跳过此被试。")
        return []

    sorted_keys = sorted(data.keys(), key=lambda x: int(x))
    for key in sorted_keys:
        captions.append(data[key])
    return captions


# --- 2. 加载 Ground Truth 数据 (只需加载一次) ---
print("正在加载 Ground Truth 字幕...")
refs1 = extract_sentences_from_json(file1_path, is_list_value=True)
print("Ground Truth 加载完毕。\n")

# --- 3. 循环处理每个被试并汇报结果 ---

# 初始化列表，用于存储所有被试的分数以计算平均值
all_scores_qwen = {'P': [], 'R': [], 'F': []}
all_scores_coco = {'P': [], 'R': [], 'F': []}
hashname_model = ""

print("========== 开始计算各被试的BERTScore ==========")
for sub in subjects:
    print(f"\n--- 正在处理被试: {sub} ---")

    # 动态构建当前被试的结果文件名
    file_name = f"{sub}_step_1800_easy_reason_reformat.json"
    file_path = os.path.join(results_dir, file_name)

    # 加载当前被试的解码字幕
    captions = load_captions_from_results(file_path)

    # 如果文件不存在或为空，则跳到下一个被试
    if not captions:
        continue

    # a. 计算并汇报 vs qwen gt 的分数
    (P, R, F), hashname = score(captions, refs1, lang="en", return_hash=True)
    if not hashname_model: hashname_model = hashname  # 记录哈希值

    p_mean, r_mean, f_mean = P.mean().item(), R.mean().item(), F.mean().item()
    print(f"  vs qwen gt: P={p_mean:.6f} R={r_mean:.6f} F={f_mean:.6f}")

    # 存储分数用于后续平均
    all_scores_qwen['P'].append(p_mean)
    all_scores_qwen['R'].append(r_mean)
    all_scores_qwen['F'].append(f_mean)



# --- 4. 计算并汇报平均值 ---
print("\n\n========== 所有被试的平均BERTScore ==========")
num_subjects_processed = len(all_scores_qwen['P'])
print(f"基于 {num_subjects_processed} 名成功处理的被试计算平均值。")

# 计算 qwen gt 的平均分
if num_subjects_processed > 0:
    avg_P_qwen = np.mean(all_scores_qwen['P'])
    avg_R_qwen = np.mean(all_scores_qwen['R'])
    avg_F_qwen = np.mean(all_scores_qwen['F'])
    print(f"\n平均分 (vs qwen gt) -> {hashname_model}:")
    print(f"  P={avg_P_qwen:.6f} R={avg_R_qwen:.6f} F={avg_F_qwen:.6f}")


else:
    print("\n未能计算平均分，因为没有成功处理任何被试的数据。")