import numpy as np
import os
import json
from tqdm import tqdm

# 1. 定义文件路径
npy_file_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub1/test_img_index_start_from0.npy'
json_folder_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA'
output_json_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/short_VQA_groundtruth.json'

# 2. 加载npy文件中的索引数据
print("正在加载索引文件...")
indices = np.load(npy_file_path)
print(f"成功加载 {len(indices)} 个索引")

# 3. 初始化结果字典
result_dict = {}

# 4. 遍历每个索引，提取对应的Answer
print("开始提取Answer内容...")
error_count = 0
error_log = []

for idx, num in tqdm(enumerate(indices), total=len(indices), desc="处理进度"):
    try:
        # 将数字转换为5位字符串（如123 -> "00123"）
        json_filename = f"{num:05d}.json"
        json_filepath = os.path.join(json_folder_path, json_filename)

        # 检查文件是否存在
        if not os.path.exists(json_filepath):
            raise FileNotFoundError(f"文件不存在")

        # 读取json文件并提取Answer
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 验证json格式
            if not isinstance(data, list) or len(data) == 0 or "Answer" not in data[0]:
                raise ValueError("JSON格式不符合要求，缺少Answer字段")

            answer = data[0]["Answer"]

            # 添加到结果字典，键为字符串格式的序号
            result_dict[str(idx)] = answer

    except Exception as e:
        error_count += 1
        error_log.append(f"索引 {num} (位置 {idx}) 处理失败: {str(e)}")

# 5. 保存结果到目标json文件
with open(output_json_path, 'w', encoding='utf-8') as f_out:
    json.dump(result_dict, f_out, indent=2, ensure_ascii=False)

# 6. 输出处理结果统计
print("\n处理完成！")
print(f"成功提取: {len(indices) - error_count} 条Answer")
print(f"处理失败: {error_count} 条")
print(f"结果已保存至: {output_json_path}")

# 打印错误详情（如果有）
if error_count > 0:
    print("\n错误详情:")
    for error in error_log:
        print(f"- {error}")
