

"""
import os
import json
from glob import glob
from tqdm import tqdm

# 目标文件夹路径（与你提供的一致）
target_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning"

# 获取所有JSON文件
json_files = glob(os.path.join(target_dir, "*.json"))
if not json_files:
    print(f"未在 {target_dir} 找到JSON文件")
else:
    print(f"找到 {len(json_files)} 个JSON文件，开始清理所有<image>标签...")

failed_files = []

for file_path in tqdm(json_files, desc="批量处理"):
    try:
        # 读取文件
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 核心修复：覆盖3种常见格式（<image>\n、\n<image>、单独<image>）
        if isinstance(data, list) and len(data) > 0 and "Question" in data[0]:
            raw_question = data[0]["Question"]
            # 按顺序替换，确保所有情况都被处理
            cleaned_question = raw_question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            data[0]["Question"] = cleaned_question

        # 写回原文件（覆盖）
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        failed_files.append({"file": file_path, "error": str(e)})

# 输出处理结果
print(f"\n处理完成！")
print(f"成功处理：{len(json_files) - len(failed_files)} 个文件")
if failed_files:
    print(f"失败：{len(failed_files)} 个文件（可查看路径排查）：")
    for fail in failed_files:
        print(f"- {fail['file']}：{fail['error']}")
else:
    print("所有文件的<image>标签已完全移除，包括开头、结尾或单独出现的情况")
"""


"""

import numpy as np
import torch
from transformers import CLIPTokenizer
import os
import json
import glob
from tqdm import tqdm

# --- 1. 初始化配置（沿用你的参数） ---
print(">>> 1. 初始化模型与路径...")
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 加载tokenizer（确保路径正确）
MODEL_PATH = "/data/home/luyizhuo/Datastation_lyz/Models/Muddit/tokenizer/"
tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH)

# 目标JSON文件夹路径
JSON_DIR = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning"


# --- 2. 定义tokenize函数（完全沿用你的实现） ---
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


# --- 3. 核心筛选逻辑：遍历文件+token长度判断+删除 ---
def filter_files_by_question_token_length(json_dir, max_allowed_tokens=16):
    # 1. 获取所有JSON文件路径
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    total_files = len(json_files)
    deleted_count = 0
    failed_count = 0  # 记录处理失败的文件（如格式错误）
    failed_files = []

    if total_files == 0:
        print(f"警告：在 {json_dir} 路径下未找到任何JSON文件")
        return 0, 0, 0

    print(f"\n>>> 2. 开始处理 {total_files} 个JSON文件，最大允许token数：{max_allowed_tokens}")

    # 2. 遍历每个文件
    for file_path in tqdm(json_files, desc="筛选文件（token数>16则删除）"):
        try:
            # 读取JSON文件并提取Question
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 检查数据格式（确保是 [{Question: "...", Answer: "..."}] 结构）
            if not (isinstance(data, list) and len(data) > 0 and "Question" in data[0]):
                raise ValueError("文件格式错误：未找到有效的Question字段")

            question = data[0]["Question"]

            # 3. Tokenize并计算长度（完全沿用你的逻辑：去掉最后一个token）
            question_ids = tokenize_prompt(
                tokenizer,
                question,
                padding=False,  # 按你的示例设为False
            )
            question_ids = question_ids[:, :-1]  # 移除最后一个token（如EOS）
            q_len = len(question_ids[0])  # 获取token长度

            # 4. 判断是否超过16，超过则删除文件
            if q_len > max_allowed_tokens:
                os.remove(file_path)
                deleted_count += 1

        except Exception as e:
            # 捕获所有异常（如文件损坏、格式错误），记录失败信息
            failed_count += 1
            failed_files.append({"文件路径": file_path, "错误原因": str(e)})

    # --- 4. 输出统计结果 ---
    print(f"\n>>> 处理完成！")
    print(f"总文件数：{total_files}")
    print(f"保留文件数：{total_files - deleted_count - failed_count}")
    print(f"删除文件数（token>16）：{deleted_count}")
    print(f"处理失败文件数（格式错误/损坏）：{failed_count}")

    # 打印失败文件详情（如需排查）
    if failed_count > 0:
        print(f"\n处理失败的文件详情：")
        for fail in failed_files:
            print(f"- {fail['文件路径']}：{fail['错误原因']}")

    return deleted_count, failed_count, total_files


# --- 5. 执行筛选并获取删除数量 ---
deleted_num, failed_num, total_num = filter_files_by_question_token_length(JSON_DIR, max_allowed_tokens=16)
print(f"\n最终：共删除 {deleted_num} 个Question token数超过16的JSON文件")

"""





import os
import json
import glob
from tqdm import tqdm

# --- 1. 定义核心路径（按你的需求配置） ---
# easy_reasoning文件夹（目标：读取并最终保存整合结果）
easy_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning"
# detail文件夹（来源：提取Answer用于整合）
detail_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/detail"


# --- 2. 核心整合逻辑 ---
def integrate_answer_from_detail_to_easy(easy_dir, detail_dir):
    # 1. 获取easy_reasoning下所有JSON文件（仅处理.json后缀）
    easy_json_files = glob.glob(os.path.join(easy_dir, "*.json"))
    total_files = len(easy_json_files)
    success_count = 0  # 成功整合的文件数
    missing_detail_count = 0  # detail中缺失对应文件的数量
    error_count = 0  # 格式错误/读取失败的文件数
    error_log = []  # 记录错误详情

    if total_files == 0:
        print(f"警告：在 {easy_dir} 未找到任何JSON文件，终止任务")
        return

    print(f"找到 {total_files} 个easy_reasoning文件，开始匹配detail文件并整合Answer...\n")

    # 2. 遍历每个easy_reasoning文件
    for easy_file_path in tqdm(easy_json_files, desc="整合进度"):
        # 获取当前文件的文件名（如"00123.json"），用于匹配detail文件夹
        file_name = os.path.basename(easy_file_path)
        # 构造detail文件夹中对应文件的路径
        detail_file_path = os.path.join(detail_dir, file_name)

        try:
            # --- 步骤1：读取easy_reasoning文件的Question和原始Answer ---
            with open(easy_file_path, "r", encoding="utf-8") as f_easy:
                easy_data = json.load(f_easy)
            # 验证easy文件格式（确保是 [{Question:..., Answer:...}] 结构）
            if not (isinstance(easy_data, list) and len(easy_data) > 0):
                raise ValueError("easy文件格式错误：非列表结构或列表为空")
            if "Question" not in easy_data[0] or "Answer" not in easy_data[0]:
                raise KeyError("easy文件缺失Question或Answer字段")

            easy_question = easy_data[0]["Question"]
            easy_original_answer = easy_data[0]["Answer"]

            # --- 步骤2：读取detail文件的Answer ---
            if not os.path.exists(detail_file_path):
                missing_detail_count += 1
                error_log.append(f"缺失detail文件：{file_name}（easy路径：{easy_file_path}）")
                continue  # 跳过缺失的文件，不终止后续处理

            with open(detail_file_path, "r", encoding="utf-8") as f_detail:
                detail_data = json.load(f_detail)
            # 验证detail文件格式
            if not (isinstance(detail_data, list) and len(detail_data) > 0 and "Answer" in detail_data[0]):
                raise ValueError("detail文件格式错误：缺失Answer字段或结构异常")

            detail_answer = detail_data[0]["Answer"]

            # --- 步骤3：按要求整合Answer ---
            # 格式：easy原始Answer + " (Thinking: detail的Answer)"
            integrated_answer = f"{easy_original_answer} (Thinking: {detail_answer})"
            # 更新easy_data中的Answer
            easy_data[0]["Answer"] = integrated_answer

            # --- 步骤4：将整合后的数据写回easy_reasoning文件（覆盖原文件） ---
            with open(easy_file_path, "w", encoding="utf-8") as f_save:
                json.dump(easy_data, f_save, indent=2, ensure_ascii=False)

            success_count += 1  # 记录成功整合的文件

        except Exception as e:
            # 捕获所有异常（如文件损坏、格式错误），避免程序崩溃
            error_count += 1
            error_log.append(f"处理失败：{file_name}（错误：{str(e)}，easy路径：{easy_file_path}）")

    # --- 3. 输出最终处理结果 ---
    print("\n" + "=" * 50)
    print("整合任务完成！结果统计：")
    print(f"1. 总处理文件数：{total_files}")
    print(f"2. 成功整合文件数：{success_count}")
    print(f"3. detail缺失文件数：{missing_detail_count}")
    print(f"4. 处理失败文件数（格式错误/损坏）：{error_count}")
    print("=" * 50)

    # 打印错误详情（如需排查问题）
    if error_log:
        print(f"\n错误详情（共{len(error_log)}条）：")
        for log in error_log:
            print(f"- {log}")


# --- 4. 执行整合任务 ---
if __name__ == "__main__":
    integrate_answer_from_detail_to_easy(easy_dir, detail_dir)
