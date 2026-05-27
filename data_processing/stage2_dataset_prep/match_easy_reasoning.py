import os
import json
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def process_nsd_to_llava_detail():
    """
    将NSD数据集中的刺激图像与LLaVA的detail_23k.json数据集中的对话数据进行匹配，
    并根据匹配结果生成相应的JSON文件或记录未匹配项。
    适配多轮对话格式，仅提取第一轮QA对。
    """
    # 1. 定义文件和文件夹路径
    # --- 基础路径 (不变) ---
    stim_order_f = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_IDs/nsd_expdesign.mat"
    coco_id_template = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_IDs/cocoId_sub{sub_id}.npy"

    # --- 路径变更点 ---
    llava_json_path = "/data/home/luyizhuo/Datastation_lyz/Datasets/LLaVA-Instruct-150k/conversation_58k.json"
    output_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning/"

    unmatched_npy_path = os.path.join(output_dir, "unmatched_nsd_indices.npy")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输入文件: {llava_json_path}")
    print(f"输出目录已确认为: {output_dir}")

    # 2. 加载并预处理LLaVA对话数据为字典以便快速查找
    print("正在加载并预处理LLaVA JSON数据 (llava_instruct_150k)...")
    llava_data_map = {}
    with open(llava_json_path, 'r', encoding='utf-8') as f:
        llava_data = json.load(f)
        for item in llava_data:
            # 提取当前item的COCO ID（转为整数，避免字符串格式不统一）
            coco_id = int(item['id'])
            conversations = item['conversations']

            question = ""
            answer = ""
            # 核心修改：仅提取多轮对话中的第一轮QA（human→gpt）
            # 1. 检查对话列表长度至少为2（保证有第一轮问答）
            # 2. 确认第一轮是human提问，第二轮是gpt回答
            if len(conversations) >= 2:
                first_turn = conversations[0]
                second_turn = conversations[1]
                if first_turn['from'] == 'human' and second_turn['from'] == 'gpt':
                    question = first_turn['value']  # 第一轮人类提问
                    answer = second_turn['value']  # 第一轮GPT回答

            # 仅当QA都非空时，存入字典（过滤无效数据）
            if question and answer:
                llava_data_map[coco_id] = {"Question": question, "Answer": answer}
    print(f"LLaVA数据加载完毕，共 {len(llava_data_map)} 条有效第一轮QA对话。")

    # 3. 加载NSD刺激顺序数据（逻辑不变）
    print("正在加载NSD刺激顺序数据...")
    stim_order = loadmat(stim_order_f)['subjectim']
    print("NSD数据加载完毕。")

    # 4. 初始化用于记录的列表和集合（逻辑不变）
    unmatched_list = []
    processed_nsd_indices = set()

    # 5. 遍历每个被试的数据进行匹配（逻辑不变）
    total_subjects = stim_order.shape[0]
    total_trials = stim_order.shape[1]

    for sub_id in range(1, total_subjects + 1):
        print(f"\n--- 正在处理被试 {sub_id}/{total_subjects} ---")

        coco_ids_path = coco_id_template.format(sub_id=sub_id)
        if not os.path.exists(coco_ids_path):
            print(f"警告: 找不到文件 {coco_ids_path}，跳过被试 {sub_id}")
            continue

        subject_coco_ids = np.load(coco_ids_path)
        subject_nsd_indices = stim_order[sub_id - 1, :]

        for i in tqdm(range(total_trials), desc=f"被试 {sub_id} 进度"):
            nsd_index = subject_nsd_indices[i]

            # 跳过已处理的NSD索引（避免重复生成文件）
            if nsd_index in processed_nsd_indices:
                continue
            processed_nsd_indices.add(nsd_index)

            # 获取当前 trial 对应的COCO ID
            coco_id = subject_coco_ids[i]

            # 匹配LLaVA数据并生成输出文件
            if coco_id in llava_data_map:
                conversation_data = llava_data_map[coco_id]
                output_data = [conversation_data]  # 保持原输出格式：列表包裹字典

                # 计算输出文件索引（NSD标号-1）并格式化为5位数字
                file_index = nsd_index - 1
                output_filename = f"{file_index:05d}.json"
                output_filepath = os.path.join(output_dir, output_filename)

                # 写入JSON文件（保持缩进和编码格式）
                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    json.dump(output_data, f_out, indent=2, ensure_ascii=False)
            else:
                # 记录未匹配的索引（NSD标号-1）
                unmatched_list.append(nsd_index - 1)

    # 6. 处理完毕，保存并打印未匹配的列表（逻辑不变）
    print("\n--- 所有被试处理完毕 ---")

    # 排序并保存未匹配索引
    unmatched_array = np.array(sorted(unmatched_list))
    np.save(unmatched_npy_path, unmatched_array)
    print(f"总共找到 {len(unmatched_array)} 个未匹配的图像。")
    print(f"未匹配的图像索引（NSD标号-1）已排序并保存至: {unmatched_npy_path}")

    print("\n未匹配的NSD索引 (NSD标号-1) 列表:")
    print(unmatched_array.tolist())

    # 统计匹配情况
    total_unique_stim = len(processed_nsd_indices)
    matched_count = total_unique_stim - len(unmatched_array)
    print(f"\n在 {total_unique_stim} 张独立刺激图像中，成功匹配 {matched_count} 张。")


if __name__ == '__main__':
    process_nsd_to_llava_detail()