import numpy as np
import torch
from transformers import CLIPTokenizer
import os
from tqdm import tqdm
import os
import json
import glob
from tqdm import tqdm



# --- 1. 设置与初始化 ---
print(">>> 1. Setting up models, paths, and parameters...")

DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_PATH = "/data/home/luyizhuo/Datastation_lyz/Models/Muddit/tokenizer/"
tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH)


# --- 2. 定义tokenize函数 ---
@torch.no_grad()
def tokenize_prompt(
        tokenizer,
        prompt,
        text_encoder_architecture='open_clip',
        padding='max_length',
        max_length=77,
):
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        input_ids = tokenizer(
            prompt,
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids
        return input_ids




def process_qa_from_json_files_with_filenames(directory_path):
    """
    扫描一个目录中的所有JSON文件，提取文件名、问题和答案。

    Args:
        directory_path (str): 存有JSON文件的目录路径。

    Returns:
        tuple: (filenames_list, questions_list, full_prompts_list)
               - filenames_list (list): 包含所有JSON文件名（不含扩展名）的列表。
               - questions_list (list): 包含所有清理后问题字符串的列表。
               - full_prompts_list (list): 包含所有 "问题 + 答案" 拼接后字符串的列表。
               这三个列表的顺序是一一对应的。
    """
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"错误: 目录未找到 '{directory_path}'")
        return [], [], []

    # 使用glob查找所有.json文件，并排序以确保顺序一致
    json_files = sorted(glob.glob(os.path.join(directory_path, '*.json')))

    if not json_files:
        print(f"警告: 在目录 '{directory_path}' 中没有找到任何JSON文件。")
        return [], [], []

    print(f"在目录中找到了 {len(json_files)} 个JSON文件，开始处理...")

    # 1. 初始化三个空列表用于存储结果
    filenames_list = []
    questions_list = []
    full_prompts_list = []

    # 遍历所有找到的JSON文件
    for file_path in tqdm(json_files, desc="正在处理JSON文件"):
        try:
            # --- 新增功能: 提取文件名 ---
            # 1. 获取完整文件名，如 "04589.json"
            base_name = os.path.basename(file_path)
            # 2. 分离文件名和扩展名，并获取文件名部分，如 "04589"
            file_name = os.path.splitext(base_name)[0]
            filenames_list.append(file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list) and len(data) > 0:
                    item = data[0]

                    # --- 任务 1: 处理问题 ---
                    raw_question = item.get("Question", "")
                    cleaned_question = raw_question.replace("<image>", "").replace("\n", "").strip()
                    questions_list.append(cleaned_question)

                    # --- 任务 2: 处理答案并拼接 ---
                    answer = item.get("Answer", "")
                    full_prompt = cleaned_question + " " + answer
                    full_prompts_list.append(full_prompt)

        except json.JSONDecodeError:
            print(f"\n警告: 文件 {os.path.basename(file_path)} 不是有效的JSON格式，已跳过。")
            # 如果文件解析失败，为了保持列表对应，也移除刚刚添加的文件名
            if filenames_list:
                filenames_list.pop()
        except Exception as e:
            print(f"\n处理文件 {os.path.basename(file_path)} 时发生未知错误: {e}，已跳过。")
            if filenames_list:
                filenames_list.pop()

    # 检查列表长度是否一致，这是一个好习惯
    assert len(filenames_list) == len(questions_list) == len(full_prompts_list), "处理后的列表长度不一致！"

    print("所有文件处理完毕！")
    return filenames_list, questions_list, full_prompts_list





filenames_list, questions_list, full_prompts_list = process_qa_from_json_files_with_filenames(directory_path='/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA/')

Q_lens = []

for i in tqdm(range(len(filenames_list))):
    file_name = filenames_list[i]
    output_path = os.path.join('/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA_token_ids/', f'{file_name}.npy')
    question = questions_list[i]
    QA = full_prompts_list[i]

    question_ids = tokenize_prompt(
        tokenizer,
        question,
        padding=False,
    )
    question_ids = question_ids[:, :-1]
    q_len = len(question_ids[0])
    Q_lens.append(q_len)

    QA_ids = tokenize_prompt(tokenizer, QA)

    QA_ids_np = QA_ids.squeeze(0).numpy()  # 移除批次维度并转换为numpy
    np.save(output_path, QA_ids_np)

np.save('/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA_Q_len.npy', np.array(Q_lens))


















