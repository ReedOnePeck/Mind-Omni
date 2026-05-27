import numpy as np
import os
import json
from tqdm import tqdm
import random

def create_caption_json_file():
    """
    根据给定的图像索引，从文本文件中提取字幕，并生成一个JSON Lines文件。
    """
    # 1. 定义文件和文件夹路径
    index_npy_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub1/test_img_index_start_from0.npy'
    captions_dir = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_captions_recapted_Qw2VL'
    output_json_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/brain2caption_qwen.json'

    print(f"开始处理...")
    print(f"加载索引文件: {index_npy_path}")

    # 2. 加载包含编号的ndarray
    try:
        image_indices = np.load(index_npy_path)
        print(f"成功加载 {len(image_indices)} 个索引。")
    except FileNotFoundError:
        print(f"错误: 索引文件未找到: {index_npy_path}")
        return

    # 3. 准备写入输出文件
    print(f"将把结果写入: {output_json_path}")

    # 使用 'w' 模式打开文件，逐行写入
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        # 使用tqdm来显示进度条
        for index in tqdm(image_indices, desc="正在提取字幕并生成JSON"):
            # 将编号转换为5位数的字符串文件名
            # 例如: 123 -> "00123.txt"
            caption_filename = f"{index:05d}.txt"
            caption_filepath = os.path.join(captions_dir, caption_filename)

            try:
                # 读取对应的txt文件内容
                with open(caption_filepath, 'r', encoding='utf-8') as f_in:
                    # 读取整行并去除首尾可能存在的空白符
                    sentence = f_in.read().strip()

                # 按照指定格式创建字典
                output_data = {"prompt": sentence}

                # 将字典转换为JSON字符串并写入文件，并在末尾添加换行符
                f_out.write(json.dumps(output_data) + '\n')

            except FileNotFoundError:
                print(f"\n警告: 未找到字幕文件: {caption_filepath}，已跳过此索引: {index}")
            except Exception as e:
                print(f"\n处理文件 {caption_filepath} 时发生错误: {e}")

    print(f"\n处理完成！已成功创建文件: {output_json_path}")


def create_raw_coco_json():
    """
    加载COCO字幕，为73k张图片各随机选择一条非空字幕，
    然后根据给定的1000个索引，提取对应的字幕并生成一个JSON Lines文件。
    """
    # 1. 定义文件和文件夹路径
    BASE_DATA_PATH = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/"
    CAPTIONS_PATH = os.path.join(BASE_DATA_PATH, 'COCO_73k_annots_curated.npy')

    # 与上一个任务相同的索引文件
    index_npy_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub1/test_img_index_start_from0.npy'

    # 新的输出文件
    output_json_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/raw_COCO.json'

    # --- 步骤 1: 预处理，生成 selected_captions 列表 ---
    print(f"正在加载字幕文件: {CAPTIONS_PATH}")
    try:
        # 加载包含嵌套列表的npy文件通常需要 allow_pickle=True
        captions_data = np.load(CAPTIONS_PATH, allow_pickle=True)
        num_samples = len(captions_data)
        print(f"找到 {num_samples} 个样本。")
    except FileNotFoundError:
        print(f"错误: 字幕文件未找到: {CAPTIONS_PATH}")
        return

    # 随机选择非空caption
    print("预处理中：为每个样本随机选择一个非空字幕...")
    selected_captions = []
    # 使用tqdm显示预处理进度
    for i in tqdm(range(num_samples), desc="预处理字幕"):
        # 筛选出当前样本中所有非空的字幕
        valid_captions = [caption for caption in captions_data[i] if caption]
        if valid_captions:
            # 如果存在非空字幕，则从中随机选择一个
            selected_captions.append(random.choice(valid_captions))
        else:
            # 如果所有字幕都是空的，则添加一个空字符串作为占位符
            selected_captions.append("")

    print(f"预处理完成。已生成 {len(selected_captions)} 条选定字幕。")

    # --- 步骤 2: 根据索引提取字幕并保存为JSON ---
    print(f"\n正在加载索引文件: {index_npy_path}")
    try:
        image_indices = np.load(index_npy_path)
        print(f"成功加载 {len(image_indices)} 个待提取的索引。")
    except FileNotFoundError:
        print(f"错误: 索引文件未找到: {index_npy_path}")
        return

    print(f"将根据索引提取字幕并写入: {output_json_path}")

    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        # 遍历1000个索引
        for index in tqdm(image_indices, desc="正在生成JSON文件"):
            # 检查索引是否有效
            if index < len(selected_captions):
                # 从预处理好的列表中获取对应的字幕
                caption = selected_captions[index]

                # 按照指定格式创建字典
                output_data = {"prompt": caption}

                # 将字典转换为JSON字符串并写入文件，添加换行符
                f_out.write(json.dumps(output_data) + '\n')
            else:
                print(f"\n警告: 索引 {index} 超出范围，已跳过。")

    print(f"\n处理完成！已成功创建文件: {output_json_path}")


if __name__ == '__main__':
    create_raw_coco_json()