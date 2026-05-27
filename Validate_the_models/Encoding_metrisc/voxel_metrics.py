import numpy as np
import os
from scipy.spatial.distance import pdist  # 引入pdist来高效计算RDM
# from sklearn.metrics import mean_squared_error # 不再需要sklearn
from scipy.stats import pearsonr  # 引入PCC计算

# --- 1. 基本设置 ---

# 定义被试列表
subjects = [1, 2, 5, 7]

# 定义数据路径
gt_base_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/'
recons_dir_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage1_2'

# 用于存储每个被试的各项指标分数
all_rdm_corr_scores = []
all_pcc_scores = []
all_mse_scores = []

print("========== 开始计算fMRI重建结果的评价指标 ==========")
print("将计算: RDM相关性(RSA), 皮尔逊相关系数(PCC), 均方误差(MSE)")

# --- 2. 循环处理每个被试 ---
for sub in subjects:
    print(f"\n--- 正在处理被试 (Subject) {sub} ---")

    try:
        # --- a. 加载 Ground Truth fMRI 数据 ---
        gt_file_path = os.path.join(gt_base_path, f'test_data_sub{sub}', f'sub{sub}_test_multi.npy')
        test_fMRI_multi = np.load(gt_file_path)
        print(f"  成功加载 Ground Truth: {os.path.basename(gt_file_path)}")

        # --- b. 自动查找并加载重建的 fMRI 数据 ---
        recons_filename = None
        for fname in os.listdir(recons_dir_path):
            if fname.startswith(f'sub{sub}') and fname.endswith('_MM.npy'):
                recons_filename = fname
                break

        if recons_filename is None:
            print(f"  >> 警告: 未能在目录中找到被试 {sub} 的重建文件。跳过此被试。")
            continue

        recons_file_path = os.path.join(recons_dir_path, recons_filename)
        recons_fMRI = np.load(recons_file_path)
        print(f"  成功加载重建结果: {recons_filename}")

        # --- c. 检查数据维度是否一致 ---
        if test_fMRI_multi.shape != recons_fMRI.shape:
            print(f"  >> 警告: 被试 {sub} 的数据维度不匹配！")
            print(f"     Ground Truth Shape: {test_fMRI_multi.shape}")
            print(f"     Reconstruction Shape: {recons_fMRI.shape}")
            print("     跳过此被试的计算。")
            continue

        num_samples = test_fMRI_multi.shape[0]

        # --- d. 计算各项指标 ---

        # (1) 计算RDM相关性 (RSA)
        print("    (1) 正在计算RDM相关性...")
        rdm_gt_vec = pdist(test_fMRI_multi, metric='correlation')
        rdm_recon_vec = pdist(recons_fMRI, metric='correlation')
        correlation_matrix = np.corrcoef(rdm_gt_vec, rdm_recon_vec)
        rdm_correlation = correlation_matrix[0, 1]
        all_rdm_corr_scores.append(rdm_correlation)
        print(f"      > RDM 相关性: {rdm_correlation:.6f}")

        # (2) 计算平均皮尔逊相关系数 (PCC)
        print("    (2) 正在计算平均PCC...")
        subject_pcc_scores = []
        for i in range(num_samples):
            # pearsonr返回相关系数和p-value，我们只需要前者
            pcc, _ = pearsonr(test_fMRI_multi[i], recons_fMRI[i])
            subject_pcc_scores.append(pcc)

        avg_pcc = np.mean(subject_pcc_scores)
        all_pcc_scores.append(avg_pcc)
        print(f"      > 平均 PCC: {avg_pcc:.6f}")

        # (3) 计算平均均方误差 (MSE) - 手动计算
        print("    (3) 正在计算平均MSE...")
        subject_mse_scores = []
        for i in range(num_samples):
            # ---【核心改动】---
            # 手动计算MSE: (真实值 - 预测值)的平方的均值
            mse = np.mean((test_fMRI_multi[i] - recons_fMRI[i]) ** 2)
            subject_mse_scores.append(mse)

        avg_mse = np.mean(subject_mse_scores)
        all_mse_scores.append(avg_mse)
        print(f"      > 平均 MSE: {avg_mse:.6f}")


    except FileNotFoundError as e:
        print(f"  >> 错误: 文件未找到。请检查路径。详细信息: {e}")
    except Exception as e:
        print(f"  >> 处理被试 {sub} 时发生未知错误: {e}")

# --- 3. 计算并打印平均结果 ---
if all_rdm_corr_scores:  # 确保列表不为空
    average_rdm_corr = np.mean(all_rdm_corr_scores)
    average_pcc = np.mean(all_pcc_scores)
    average_mse = np.mean(all_mse_scores)

    print("\n\n========== 最终结果总结 ==========")
    print(f"成功处理了 {len(all_rdm_corr_scores)}/{len(subjects)} 名被试。")
    print(f"  - 所有被试的平均 RDM 相关性为: {average_rdm_corr:.6f}")
    print(f"  - 所有被试的平均 PCC 为: {average_pcc:.6f}")
    print(f"  - 所有被试的平均 MSE 为: {average_mse:.6f}")
else:
    print("\n\n未能成功计算任何被试的评价指标。")