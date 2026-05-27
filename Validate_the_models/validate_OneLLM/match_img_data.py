import numpy as np
import torch
from tqdm import tqdm
import os


def match_features_by_clip():
    """
    使用预先提取的CLIP特征，通过计算余弦相似度来匹配两组图像，并保存结果。
    """
    # --- 1. 定义文件路径 ---
    # 输入特征文件
    features_982_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/982_img_CLIP.npy'
    features_1000_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/1000_img_CLIP.npy'

    # 1000张图像的原始索引文件，用于生成文件名
    indices_1000_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub1/test_img_index_start_from0.npy'

    # 输出文件路径
    output_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/matched_names_by_CLIP.npy'

    # --- 2. 加载数据 ---
    print("正在加载CLIP特征和索引文件...")
    try:
        features_982 = np.load(features_982_path)
        features_1000 = np.load(features_1000_path)
        indices_1000 = np.load(indices_1000_path)
    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e}")
        print("请确保所有输入文件路径都正确无误。")
        return

    print(f"已加载982个特征，形状: {features_982.shape}")
    print(f"已加载1000个特征，形状: {features_1000.shape}")
    print(f"已加载1000个索引，形状: {indices_1000.shape}")

    # --- 3. 计算余弦相似度 ---
    # 为了高效计算，我们使用PyTorch，并尽可能利用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n将使用设备: {device} 进行计算")

    # 将Numpy数组转换为PyTorch张量
    features_982_tensor = torch.from_numpy(features_982).to(device, dtype=torch.float32)
    features_1000_tensor = torch.from_numpy(features_1000).to(device, dtype=torch.float32)

    # L2归一化向量。归一化后，向量的点积就等于它们的余弦相似度。
    # A · B / (||A|| * ||B||) -> A_norm · B_norm
    features_982_tensor = features_982_tensor / features_982_tensor.norm(dim=1, keepdim=True)
    features_1000_tensor = features_1000_tensor / features_1000_tensor.norm(dim=1, keepdim=True)

    print("正在计算相似度矩阵...")
    # 使用矩阵乘法一次性计算所有对的相似度
    # (982, D) @ (D, 1000) -> (982, 1000)
    similarity_matrix = torch.matmul(features_982_tensor, features_1000_tensor.T)
    print(f"相似度矩阵计算完成，形状: {similarity_matrix.shape}")

    # --- 4. 寻找最佳匹配 ---
    print("正在为982个特征寻找最佳匹配...")
    # 对矩阵的每一行，找到最大值的索引。这就是最佳匹配项在1000个特征中的位置。
    best_match_indices = torch.argmax(similarity_matrix, dim=1)

    # 将结果从GPU移回CPU，并转换为Numpy数组
    best_match_indices = best_match_indices.cpu().numpy()

    # --- 5. 生成文件名并保存结果 ---
    # 使用找到的索引从原始索引数组中查找图像编号
    matched_original_indices = indices_1000[best_match_indices]

    # 将图像编号格式化为文件名
    matched_filenames = [f"{index:05d}.png" for index in matched_original_indices]

    # 验证我们是否得到了982个文件名
    if len(matched_filenames) != features_982.shape[0]:
        print(f"错误：匹配到的文件名数量 ({len(matched_filenames)}) 与目标数量 ({features_982.shape[0]}) 不符！")
        return

    # 将结果列表转换为Numpy数组以便保存
    output_array = np.array(matched_filenames, dtype=object)

    try:
        np.save(output_path, output_array)
        print(f"\n成功！匹配结果已保存至:")
        print(output_path)

        # 打印一些示例以供验证
        print("\n匹配结果示例 (目标图像索引 -> 匹配到的文件名):")
        for i in range(min(10, len(output_array))):
            print(f"  目标 {i}  ->  {output_array[i]}")

    except Exception as e:
        print(f"\n错误：保存文件时发生异常 - {e}")


if __name__ == '__main__':
    match_features_by_clip()