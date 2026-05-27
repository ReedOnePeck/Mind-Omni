import os
import json
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def process_nsd_to_llava_detail():
    """
    将NSD数据集中的刺激图像与LLaVA的detail_23k.json数据集中的对话数据进行匹配，
    并根据匹配结果生成相应的JSON文件或记录未匹配项。
    """
    # 1. 定义文件和文件夹路径
    # --- 基础路径 (不变) ---
    stim_order_f = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_IDs/nsd_expdesign.mat"
    coco_id_template = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_IDs/cocoId_sub{sub_id}.npy"

    # --- 路径变更点 ---
    # (1) 更新LLaVA输入文件路径
    llava_json_path = "/data/home/luyizhuo/Datastation_lyz/Datasets/LLaVA-Instruct-150k/detail_23k.json"
    # (2) 更新输出目录路径
    output_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/raw_data/detail/"

    # 未匹配文件路径将自动使用新的输出目录
    unmatched_npy_path = os.path.join(output_dir, "unmatched_nsd_indices.npy")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输入文件: {llava_json_path}")
    print(f"输出目录已确认为: {output_dir}")

    # 2. 加载并预处理LLaVA对话数据为字典以便快速查找
    print("正在加载并预处理LLaVA JSON数据 (detail_23k)...")
    llava_data_map = {}
    with open(llava_json_path, 'r', encoding='utf-8') as f:
        llava_data = json.load(f)
        for item in llava_data:
            coco_id = int(item['id'])
            conversation = item['conversations']
            question = ""
            answer = ""
            if len(conversation) >= 2 and conversation[0]['from'] == 'human' and conversation[1]['from'] == 'gpt':
                question = conversation[0]['value']
                answer = conversation[1]['value']

            if question and answer:
                llava_data_map[coco_id] = {"Question": question, "Answer": answer}
    print(f"LLaVA数据加载完毕，共 {len(llava_data_map)} 条有效对话。")

    # 3. 加载NSD刺激顺序数据
    print("正在加载NSD刺激顺序数据...")
    stim_order = loadmat(stim_order_f)['subjectim']
    print("NSD数据加载完毕。")

    # 4. 初始化用于记录的列表和集合
    unmatched_list = []
    processed_nsd_indices = set()

    # 5. 遍历每个被试的数据进行匹配
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

            if nsd_index in processed_nsd_indices:
                continue

            processed_nsd_indices.add(nsd_index)

            coco_id = subject_coco_ids[i]

            if coco_id in llava_data_map:
                conversation_data = llava_data_map[coco_id]
                output_data = [conversation_data]

                file_index = nsd_index - 1
                output_filename = f"{file_index:05d}.json"
                output_filepath = os.path.join(output_dir, output_filename)

                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    json.dump(output_data, f_out, indent=2, ensure_ascii=False)
            else:
                unmatched_list.append(nsd_index - 1)

    # 6. 处理完毕，保存并打印未匹配的列表
    print("\n--- 所有被试处理完毕 ---")

    unmatched_array = np.array(sorted(unmatched_list))
    np.save(unmatched_npy_path, unmatched_array)
    print(f"总共找到 {len(unmatched_array)} 个未匹配的图像。")
    print(f"未匹配的图像索引（NSD标号-1）已排序并保存至: {unmatched_npy_path}")

    print("\n未匹配的NSD索引 (NSD标号-1) 列表:")
    print(unmatched_array.tolist())

    total_unique_stim = len(processed_nsd_indices)
    matched_count = total_unique_stim - len(unmatched_array)
    print(f"\n在 {total_unique_stim} 张独立刺激图像中，成功匹配 {matched_count} 张。")


if __name__ == '__main__':
    process_nsd_to_llava_detail()