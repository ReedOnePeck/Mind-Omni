import numpy as np
import os

# --- 1. 基本设置 ---

# 定义被试列表
subjects = [1, 2, 5, 7]

# 定义数据路径
gt_base_path = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/'
recons_dir_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage3'

# 用于存储每个被试的MSE分数
all_mse_scores = []

print("========== 开始计算fMRI重建结果的MSE ==========")

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
        # 遍历结果目录中的所有文件
        for fname in os.listdir(recons_dir_path):
            # 匹配以 'subX' 开头并以 '_MM.npy' 结尾的文件
            if fname.startswith(f'sub{sub}') and fname.endswith('_only_img.npy'):
                recons_filename = fname
                break  # 找到后即退出循环

        # 如果没有找到文件，则发出警告并跳到下一个被试
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
            print("     跳过此被试的MSE计算。")
            continue

        # --- d. 计算 MSE ---
        # MSE = mean((true_values - predicted_values)^2)
        mse_score = np.mean(np.square(test_fMRI_multi - recons_fMRI))

        # 将当前被试的分数存入列表
        all_mse_scores.append(mse_score)

        # 打印当前被试的结果，保留6位小数
        print(f"  >>> 被试 {sub} 的 MSE 结果: {mse_score:.6f}")

    except FileNotFoundError as e:
        print(f"  >> 错误: 文件未找到。请检查路径。详细信息: {e}")
    except Exception as e:
        print(f"  >> 处理被试 {sub} 时发生未知错误: {e}")

# --- 3. 计算并打印平均结果 ---
if all_mse_scores:  # 确保列表不为空
    average_mse = np.mean(all_mse_scores)
    print("\n\n========== 最终结果总结 ==========")
    print(f"成功处理了 {len(all_mse_scores)}/{len(subjects)} 名被试。")
    print(f"所有被试的平均 MSE 为: {average_mse:.6f}")
else:
    print("\n\n未能成功计算任何被试的MSE。")