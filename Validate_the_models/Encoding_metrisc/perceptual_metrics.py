# 导入必要的库
import torch
import torch.nn as nn
import numpy as np
import os
from scipy.spatial.distance import pdist


from train_decoder_for_perception.fMRI_recons_perceptual import fMRI_recons_perceptron


# --- 1. 初始化模型和加载权重 ---

# 设置计算设备（如果有多张GPU，请确保 "cuda:5" 是您想用的那张）
device = "cuda:5"
print(f"使用的计算设备: {device}")

# 实例化模型
model = fMRI_recons_perceptron(
    input_dim=16127,
    output_dim1=1024,
    output_dim2=29 * 1024,
    hidden_dims=[4096, 4096, 4096, 4096]
)

# 加载预训练的模型权重
try:
    checkpoint = torch.load(
        '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_perceptron/coarse_and_fine/checkpoint_epoch_40.pth',
        map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    print("模型权重加载成功！")
except Exception as e:
    print(f"错误: 模型权重加载失败: {e}")
    exit()  # 如果模型加载失败，则退出程序

# 将模型移动到指定设备，并设置为评估模式（这会关闭dropout等）
model.to(device)
model.eval()

# --- 2. 加载 Ground Truth CLIP 特征 ---

# 加载完整的CLIP特征库
CLIP_feature = np.load(
    '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_features/CLIP_feature_1024/img/img_CLIP_H_feature_1024.npy')
# 加载测试集对应的图像索引
test_index = np.load(
    '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub1/test_img_index_start_from0.npy')
# 根据索引，从特征库中提取出测试集对应的CLIP特征，作为我们的"黄金标准"
test_CLIP_feature = CLIP_feature[test_index]
print(f"Ground Truth CLIP 特征加载成功，形状为: {test_CLIP_feature.shape}")

# ==============================================================================
# 第二部分: 循环评估每个被试的重建结果
# ==============================================================================

# --- 3. 定义被试列表和数据路径 ---
subjects = [1, 2, 5, 7]
recons_dir_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage1_2'

# --- 4. 初始化用于存储所有被试分数的列表 ---
all_pcc_scores = []
all_mse_scores = []
all_rsa_scores = []

print("\n========== 开始计算重建fMRI恢复出的感知特征的评估指标 ==========")

# --- 5. 遍历每个被试 ---
for sub in subjects:
    print(f"\n--- 正在处理被试 (Subject) {sub} ---")

    try:
        # --- a. 自动查找并加载当前被试的重建fMRI数据 ---
        recons_filename = None
        # 遍历目录，找到符合命名规则的文件
        for fname in os.listdir(recons_dir_path):
            if fname.startswith(f'sub{sub}') and fname.endswith('_MM.npy'):
                recons_filename = fname
                break  # 找到后立即退出循环

        # 如果未找到文件，则打印警告并跳到下一个被试
        if recons_filename is None:
            print(f"  >> 警告: 未能在目录中找到被试 {sub} 的重建文件。跳过此被试。")
            continue

        recons_file_path = os.path.join(recons_dir_path, recons_filename)
        recon_fmri_np = np.load(recons_file_path)
        print(f"  成功加载重建的fMRI数据: {recons_filename}")

        # --- b. 使用MLP从重建fMRI中预测感知特征 ---
        print("    正在通过MLP从fMRI预测感知特征...")
        # 在 `torch.no_grad()` 环境下进行推理，以节省计算资源
        with torch.no_grad():
            # 将Numpy数组转换为PyTorch张量，并移动到GPU
            recon_fmri_tensor = torch.from_numpy(recon_fmri_np).float().to(device)

            # 模型前向传播，得到预测的特征
            predicted_features_tensor,_ = model(recon_fmri_tensor)

            # 将结果从GPU转回CPU，并转换为Numpy数组，以便后续计算
            predicted_features_np = predicted_features_tensor.cpu().numpy()

        # --- c. 检查预测特征和真实CLIP特征的维度是否一致 ---
        if test_CLIP_feature.shape != predicted_features_np.shape:
            print(f"  >> 警告: 维度不匹配！真实CLIP特征: {test_CLIP_feature.shape}, 预测特征: {predicted_features_np.shape}. 跳过。")
            continue

        # --- d. 在特征层面计算三个评估指标 ---
        print("    正在计算评估指标 (PCC, MSE, RSA)...")

        # 指标1: PCC (逐样本皮尔逊相关系数的平均值)
        # 计算1000个样本中，每个样本的真实特征向量与预测特征向量之间的相关性，然后求平均
        pcc_per_sample = [np.corrcoef(test_CLIP_feature[i], predicted_features_np[i])[0, 1] for i in
                          range(test_CLIP_feature.shape[0])]
        pcc_score = np.mean(pcc_per_sample)
        all_pcc_scores.append(pcc_score)

        # 指标2: MSE (均方误差)
        mse_score = np.mean(np.square(test_CLIP_feature - predicted_features_np))
        all_mse_scores.append(mse_score)

        # 指标3: RSA (表征相似性分析)
        # 分别计算真实特征和预测特征的RDM，然后计算两个RDM之间的相关性
        rdm_gt_vec = pdist(test_CLIP_feature, metric='correlation')
        rdm_recon_vec = pdist(predicted_features_np, metric='correlation')
        rsa_score = np.corrcoef(rdm_gt_vec, rdm_recon_vec)[0, 1]
        all_rsa_scores.append(rsa_score)

        # 打印当前被试的计算结果
        print(f"  >>> 被试 {sub} 结果: PCC={pcc_score:.6f}, MSE={mse_score:.6f}, RSA={rsa_score:.6f}")

    except FileNotFoundError as e:
        print(f"  >> 错误: 文件未找到。详细信息: {e}")
    except Exception as e:
        print(f"  >> 处理被试 {sub} 时发生未知错误: {e}")

# ==============================================================================
# 第三部分: 汇总并打印最终平均结果
# ==============================================================================
if all_pcc_scores:  # 仅当至少有一个被试成功处理时才计算平均值
    # 计算所有被试的平均分
    avg_pcc = np.mean(all_pcc_scores)
    avg_mse = np.mean(all_mse_scores)
    avg_rsa = np.mean(all_rsa_scores)

    print("\n\n========== 最终结果总结 ==========")
    print(f"成功处理了 {len(all_pcc_scores)}/{len(subjects)} 名被试。")
    print(f"所有被试的平均 PCC (越高越好): {avg_pcc:.6f}")
    print(f"所有被试的平均 MSE (越低越好): {avg_mse:.6f}")
    print(f"所有被试的平均 RSA (越高越好): {avg_rsa:.6f}")
else:
    print("\n\n未能成功计算任何被试的评估指标。")