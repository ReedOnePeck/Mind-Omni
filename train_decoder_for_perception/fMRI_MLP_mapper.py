import torch
import torch.nn as nn
from typing import List
from functools import partial


class fMRI_perceptron(nn.Module):
    """
    一个带有残差连接的多层感知机 (MLP) 模型。

    该模型首先将输入投影到一个高维空间，然后通过一系列残差块进行深度处理，
    最终投影到输出维度。

    Args:
        input_dim (int): 输入特征的维度 (例如, fMRI 的 N_voxel)。
        output_dim (int): 输出特征的维度 (例如, CLIP 特征的维度, 1024)。
        hidden_dims (List[int]): 一个包含每个残差块维度的列表。
                                 列表的长度决定了残差块的数量 (n_blocks)。
                                 例如: [4096, 4096, 4096] 表示3个隐藏维度为4096的残差块。
        activation (nn.Module, optional): 要使用的激活函数类。
                                          默认为 nn.GELU。
        norm_type (str, optional): 归一化类型, 'ln' (LayerNorm) 或 'bn' (BatchNorm)。
                                   默认为 'ln'。
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            activation: nn.Module = nn.GELU,
            norm_type: str = 'ln'
    ):
        super().__init__()

        # --- 参数验证和设置 ---
        if not hidden_dims:
            raise ValueError("hidden_dims list cannot be empty for a residual MLP.")

        # 残差块的数量由 hidden_dims 列表的长度决定
        self.n_blocks = len(hidden_dims)

        # --- 动态选择归一化和激活函数 ---
        # 这种设计借鉴了 BrainNetwork 的灵活性
        if norm_type == 'bn':
            # BatchNorm 需要知道特征数量
            norm_func = lambda h_dim: nn.BatchNorm1d(num_features=h_dim)
            # ReLU 通常与 BatchNorm 搭配
            act_fn = nn.ReLU
        elif norm_type == 'ln':
            # LayerNorm 需要知道归一化的形状
            norm_func = lambda h_dim: nn.LayerNorm(normalized_shape=h_dim)
            # GELU 通常与 LayerNorm 和 Transformer 架构搭配
            act_fn = activation  # 使用传入的 activation
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Choose 'ln' or 'bn'.")

        # --- 网络层定义 ---

        # 1. 输入层 (Input Projection Layer)
        #    将 fMRI 从 in_dim 投影到第一个残差块的维度
        first_hidden_dim = hidden_dims[0]
        self.lin0 = nn.Sequential(
            nn.Linear(input_dim, 8192),
            norm_func(8192),
            act_fn()
        )

        self.lin0_ = nn.Sequential(
            nn.Linear(8192, first_hidden_dim),
            norm_func(first_hidden_dim),
            act_fn()
        )

        # 2. 残差块 (Residual Blocks)
        self.mlp = nn.ModuleList()
        current_dim = first_hidden_dim
        for h_dim in hidden_dims:
            # 添加一个线性层来处理可能的维度变化
            self.mlp.append(nn.Sequential(
                nn.Linear(current_dim, h_dim),
                norm_func(h_dim),
                act_fn()
            ))
            current_dim = h_dim  # 更新当前维度为下一个块的输入

        # 3. 输出层 (Output Projection Layer)
        #    将最后一个残差块的输出投影到最终的 output_dim
        last_hidden_dim = hidden_dims[-1]
        self.lin1 = nn.Linear(last_hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播，包含残差连接。
        """
        # 1. 通过输入层
        x = self.lin0(x)

        x = self.lin0_(x)

        # 2. 依次通过所有残差块
        residual = x
        for i in range(self.n_blocks):
            # 获取块的输出
            x = self.mlp[i](x)

            # 添加残差连接
            # 如果维度不匹配，残差连接会通过广播（如果可能）或报错。
            # 一个更健壮的设计是为残差添加一个单独的投影层，但这里为了简单，
            # 推荐 hidden_dims 列表中的维度保持一致。
            if x.shape == residual.shape:
                x = x + residual

            # 更新残差，用于下一个块
            residual = x

        # 3. 通过输出层
        x = self.lin1(x)

        return x


# =====================================================================
# 执行样例：验证代码的正确性
# =====================================================================
if __name__ == '__main__':
    # --- 1. 定义超参数 ---
    BATCH_SIZE = 16
    N_VOXEL = 15724  # fMRI 数据的输入维度
    CLIP_DIM = 768  # CLIP 特征的输出维度

    # --- 2. 配置 MLP 结构 ---
    # 示例：创建3个残差块，每个块的隐藏维度都是 4096
    # 结构: 15724 -> 4096 -> ResBlock(4096) -> ResBlock(4096) -> ResBlock(4096) -> 768
    hidden_layer_dims = [4096, 4096, 4096]

    print(f"--- 创建带残差连接的 MLP 模型 ---")
    print(f"输入维度: {N_VOXEL}")
    print(f"残差块数量: {len(hidden_layer_dims)}")
    print(f"隐藏层维度: {hidden_layer_dims}")
    print(f"输出维度: {CLIP_DIM}")

    # --- 3. 实例化模型 ---
    residual_mlp_model = fMRI_perceptron(
        input_dim=N_VOXEL,
        output_dim=CLIP_DIM,
        hidden_dims=hidden_layer_dims,
        activation=nn.GELU,
        norm_type='ln'
    )

    # 打印模型结构
    print("\n--- 模型结构 ---")
    print(residual_mlp_model)

    # --- 4. 准备模拟输入数据 ---
    dummy_fmri_input = torch.randn(BATCH_SIZE, N_VOXEL)
    print(f"\n--- 运行测试 ---")
    print(f"模拟输入数据形状: {dummy_fmri_input.shape}")

    # --- 5. 执行前向传播 ---
    residual_mlp_model.eval()
    with torch.no_grad():
        predicted_clip_features = residual_mlp_model(dummy_fmri_input)

    # --- 6. 验证输出 ---
    print(f"模型输出数据形状: {predicted_clip_features.shape}")

    expected_shape = (BATCH_SIZE, CLIP_DIM)
    assert predicted_clip_features.shape == expected_shape, \
        f"输出形状错误！期望得到 {expected_shape}, 但实际得到 {predicted_clip_features.shape}"

    print("\n模型测试通过！输出形状正确。")

    # --- 另一个例子：维度变化的残差块 ---
    print("\n--- 创建一个维度变化的 MLP 模型 ---")
    # 结构: 15724 -> 4096 -> ResBlock(2048) -> 768
    # 注意：这种情况下，残差连接会因为维度不匹配而被跳过。
    # 这是一个设计上的选择，有时也这样用。
    varied_dims_model = fMRI_perceptron(N_VOXEL, CLIP_DIM, hidden_dims=[4096, 2048])
    print(varied_dims_model)
    output = varied_dims_model(dummy_fmri_input)
    print(f"维度变化模型输出形状: {output.shape}")
    assert output.shape == (BATCH_SIZE, CLIP_DIM)
    print("维度变化模型测试通过！")