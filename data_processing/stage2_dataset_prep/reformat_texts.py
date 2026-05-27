import os
import json
from tqdm import tqdm

# --- 1. 定义核心路径 ---
# 输入JSONL文件路径
input_jsonl_path = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage3/sub1_step_1800_easy_reason_shuffle.jsonl"
# 输出JSON文件路径（含目标文件夹）
output_dir = "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage1_2_32_32/reformated"
output_json_path = os.path.join(output_dir, "sub1_step_1800_easy_reason_shuffle.jsonl")


# --- 2. 核心转换逻辑 ---
def convert_jsonl_to_target_format(input_path, output_path):
    # 确保输出文件夹存在（不存在则创建）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 初始化目标格式字典（键为"0""1""2"...，值为含prompt的列表）
    target_dict = {}
    line_count = 0  # 记录有效行数（即键的递增序号）
    error_lines = []  # 记录解析失败的行号和内容

    # 读取JSONL文件并逐行解析
    print(f"开始读取并解析JSONL文件：{input_path}")
    with open(input_path, "r", encoding="utf-8") as f_in:
        # 用tqdm显示读取进度（针对大文件更直观）
        for line_num, line in tqdm(enumerate(f_in, start=1), desc="解析JSONL行"):
            line = line.strip()  # 去除行首尾空白（避免空行或换行符干扰）
            if not line:
                continue  # 跳过空行

            try:
                # 解析当前行的JSON数据
                json_data = json.loads(line)
                # 提取"prompt"字段（若缺失则视为错误）
                if "prompt" not in json_data:
                    raise KeyError("缺失'prompt'字段")

                prompt_content = json_data["prompt"]
                # 按目标格式添加到字典：键为字符串格式的序号，值为含prompt的列表
                target_dict[str(line_count)] = prompt_content
                line_count += 1

            except Exception as e:
                # 记录解析失败的行（行号、内容、错误原因）
                error_lines.append({
                    "line_num": line_num,
                    "line_content": line,
                    "error": str(e)
                })

    # --- 3. 将转换后的字典写入目标JSON文件 ---
    with open(output_path, "w", encoding="utf-8") as f_out:
        # 用indent=2保证JSON格式缩进清晰（与你给的示例一致）
        json.dump(target_dict, f_out, indent=2, ensure_ascii=False)

    # --- 4. 输出处理结果统计 ---
    print(f"\n格式转换完成！")
    print(f"1. 有效数据条数：{line_count}（对应目标JSON的键0~{line_count - 1}）")
    print(f"2. 输出文件路径：{output_path}")
    print(f"3. 解析失败行数：{len(error_lines)}")

    # 若有失败行，打印详情（方便排查问题）
    if error_lines:
        print(f"\n解析失败的行详情：")
        for fail in error_lines:
            print(f"- 行{fail['line_num']}：内容='{fail['line_content']}'，错误={fail['error']}")


# --- 5. 执行转换任务 ---
if __name__ == "__main__":
    convert_jsonl_to_target_format(input_jsonl_path, output_json_path)