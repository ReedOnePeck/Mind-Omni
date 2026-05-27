import os
import json
from tqdm import tqdm

# --- 1. 定义源路径和目标路径 ---
source_directory = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/short_COCO_caption/COCO_captions"
destination_directory = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA"

# --- 2. 定义固定的问题和文件总数 ---
fixed_question = "Give me a very very short description of the scene."
total_files = 73000

# --- 3. 确保目标文件夹存在，如果不存在则创建 ---
print(f"检查并确保目标目录存在: {destination_directory}")
os.makedirs(destination_directory, exist_ok=True)

print(f"开始处理 {total_files} 个文件...")

# --- 4. 循环处理每一个文件 ---
# 使用 tqdm 创建一个进度条，方便监控进度
for i in tqdm(range(total_files), desc="转换进度"):

    # --- a. 构建源文件和目标文件的完整路径 ---
    # 将数字索引格式化为五位数（例如, 123 -> "00123"）
    base_filename = f"{i:05d}"
    source_txt_path = os.path.join(source_directory, f"{base_filename}.txt")
    destination_json_path = os.path.join(destination_directory, f"{base_filename}.json")

    try:
        # --- b. 读取源txt文件的内容 ---
        # 使用 'with' 语句可以确保文件被正确关闭
        with open(source_txt_path, 'r', encoding='utf-8') as f_in:
            # .strip() 用于移除可能存在于句子前后的多余空格或换行符
            answer_text = f_in.read().strip()

        # --- c. 构建目标JSON数据结构 ---
        output_data = [
            {
                "Question": fixed_question,
                "Answer": answer_text
            }
        ]

        # --- d. 将构建好的数据写入新的JSON文件 ---
        with open(destination_json_path, 'w', encoding='utf-8') as f_out:
            # indent=2 使生成的JSON文件格式化，更易于阅读
            # ensure_ascii=False 确保非英文字符能被正确处理
            json.dump(output_data, f_out, indent=2, ensure_ascii=False)

    except FileNotFoundError:
        # 如果某个源文件不存在，打印警告并跳过
        print(f"\n警告: 源文件未找到: {source_txt_path}。已跳过。")
        continue
    except Exception as e:
        # 捕获其他可能的错误
        print(f"\n处理文件 {source_txt_path} 时发生未知错误: {e}")
        continue

# --- 5. 打印最终完成信息 ---
print("\n处理完成！")
print(f"所有 {total_files} 个文件已成功转换为JSON格式并保存至:")
print(destination_directory)