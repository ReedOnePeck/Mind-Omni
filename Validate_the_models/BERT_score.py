from bert_score import score
import json
import os
import numpy as np

# --- 1. 基本设置 (保持不变) ---
file1_path = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/brain2caption_qwen_formatted_lowercase.json"
file2_path = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/raw_COCO_formatted.json"

match = np.load('/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/982_index.npy')

# --- 2. 文件夹和被试列表 ---
parent_dir = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models"
subfolders = ["braincap", "sdrecon", "umbrae"]
subjects = ["sub01", "sub02", "sub05", "sub07"]  # 定义需要处理的四个被试


# --- 3. 函数定义 (保持不变) ---
def extract_sentences_from_json(file_path, is_list_value=False):
    """从JSON文件提取句子并按数字键排序"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sorted_keys = sorted(data.keys(), key=lambda x: int(x))
    for key in sorted_keys:
        value = data[key]
        if is_list_value:
            sentences.append(value[0])
        else:
            sentences.append(value)
    return sentences


def extract_captions_from_json(file_path):
    """读取单个JSON文件，按数字键排序提取句子"""
    captions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"警告：文件 {file_path} 未找到，将跳过此被试")
        return []
    except json.JSONDecodeError:
        print(f"警告：文件 {file_path} 格式错误，将跳过此被试")
        return []

    sorted_keys = sorted(data.keys(), key=lambda x: int(x))
    for key in sorted_keys:
        captions.append(data[key])
    return captions


# --- 4. 数据加载 (保持不变) ---
# 假设所有被试的基准字幕(Ground Truth)是相同的
print("正在加载基准字幕(Ground Truth)...")
refs1 = np.array(extract_sentences_from_json(file1_path, is_list_value=True))[match].tolist()
refs2 = np.array(extract_sentences_from_json(file2_path, is_list_value=False))[match].tolist()
print("基准字幕加载完毕。\n")

# --- 5. 主循环：计算并汇报平均分 ---
# 外层循环遍历每个模型文件夹
for folder in subfolders:
    print(f"========== 开始处理模型: {folder} ==========")

    # 初始化列表，用于存储该模型下所有被试的分数
    scores_qwen = {'P': [], 'R': [], 'F': []}
    scores_coco = {'P': [], 'R': [], 'F': []}
    hashname = ""

    # 内层循环遍历每个被试
    for sub in subjects:
        print(f"--- 正在处理被试: {sub} ---")

        # 动态构建每个被试的文件路径
        target_file = f"{sub}_decoded_caption.json"
        file_path = os.path.join(parent_dir, folder, target_file)

        # 提取当前被试的解码字幕
        candidates = extract_captions_from_json(file_path)

        # 如果文件不存在或为空，则跳过当前被试
        if not candidates:
            continue

        # 计算与 qwen gt 的分数
        (P, R, F), hashname_run = score(candidates, refs1, lang="en", return_hash=True)
        if not hashname: hashname = hashname_run  # 仅记录一次哈希名
        print(f"  vs qwen gt: P={P.mean().item():.4f} R={R.mean().item():.4f} F={F.mean().item():.4f}")
        scores_qwen['P'].append(P.mean().item())
        scores_qwen['R'].append(R.mean().item())
        scores_qwen['F'].append(F.mean().item())

        # 计算与 COCO gt 的分数
        (P, R, F), _ = score(candidates, refs2, lang="en", return_hash=True)
        print(f"  vs COCO gt: P={P.mean().item():.4f} R={R.mean().item():.4f} F={F.mean().item():.4f}")
        scores_coco['P'].append(P.mean().item())
        scores_coco['R'].append(R.mean().item())
        scores_coco['F'].append(F.mean().item())

    # --- 6. 计算并汇报当前模型的平均结果 ---
    print(f"\n--- 模型 [{folder}] 的平均结果 ({len(subjects)}名被试) ---")

    # 计算 qwen gt 的平均分
    if scores_qwen['P']:  # 确保至少有一个被试被成功处理
        avg_P = np.mean(scores_qwen['P'])
        avg_R = np.mean(scores_qwen['R'])
        avg_F = np.mean(scores_qwen['F'])
        print(f"平均分 (vs qwen gt) -> {hashname}: P={avg_P:.6f} R={avg_R:.6f} F={avg_F:.6f}")
    else:
        print("未能计算 vs qwen gt 的平均分 (无有效数据)")

    # 计算 COCO gt 的平均分
    if scores_coco['P']:  # 确保至少有一个被试被成功处理
        avg_P = np.mean(scores_coco['P'])
        avg_R = np.mean(scores_coco['R'])
        avg_F = np.mean(scores_coco['F'])
        print(f"平均分 (vs COCO gt) -> {hashname}: P={avg_P:.6f} R={avg_R:.6f} F={avg_F:.6f}")
    else:
        print("未能计算 vs COCO gt 的平均分 (无有效数据)")

    print(f"========== 模型: {folder} 处理完毕 ==========\n")