import numpy as np
import os
from scipy.spatial.distance import pdist  # 引入pdist来高效计算RDM

# --- 1. 基本设置 ---

# 定义被试列表
subjects = [1, 2, 5, 7]

# 定义数据路径
gt_base_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/'
recons_dir_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage3'

# 用于存储每个被试的RDM相关性分数
all_rdm_corr_scores = []

print("========== 开始计算fMRI重建结果的RDM相关性 ==========")

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

        # --- d. 计算RDM相关性 (替换原来的MSE计算) ---
        print("    正在计算RDM...")
        # 1. 使用 pdist 计算 Ground Truth 的 RDM (返回压缩后的上三角向量)
        # metric='correlation' 计算的是 1 - Pearson correlation
        rdm_gt_vec = pdist(test_fMRI_multi, metric='correlation')

        # 2. 计算重建结果的 RDM
        rdm_recon_vec = pdist(recons_fMRI, metric='correlation')

        # 3. 计算两个RDM向量之间的皮尔逊相关系数
        # np.corrcoef 返回一个2x2的相关矩阵，我们需要的是非对角线上的值
        correlation_matrix = np.corrcoef(rdm_gt_vec, rdm_recon_vec)
        rdm_correlation = correlation_matrix[0, 1]

        # 将当前被试的分数存入列表
        all_rdm_corr_scores.append(rdm_correlation)

        # 打印当前被试的结果
        print(f"  >>> 被试 {sub} 的 RDM 相关性: {rdm_correlation:.6f}")

    except FileNotFoundError as e:
        print(f"  >> 错误: 文件未找到。请检查路径。详细信息: {e}")
    except Exception as e:
        print(f"  >> 处理被试 {sub} 时发生未知错误: {e}")

# --- 3. 计算并打印平均结果 ---
if all_rdm_corr_scores:  # 确保列表不为空
    average_rdm_corr = np.mean(all_rdm_corr_scores)
    print("\n\n========== 最终结果总结 ==========")
    print(f"成功处理了 {len(all_rdm_corr_scores)}/{len(subjects)} 名被试。")
    print(f"所有被试的平均 RDM 相关性为: {average_rdm_corr:.6f}")
else:
    print("\n\n未能成功计算任何被试的RDM相关性。")