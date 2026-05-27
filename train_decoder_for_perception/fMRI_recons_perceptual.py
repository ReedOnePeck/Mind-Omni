import torch
import torch.nn as nn
from typing import List
from functools import partial


class fMRI_recons_perceptron(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim1: int,
            output_dim2: int,
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
        self.lin1 = nn.Linear(last_hidden_dim, output_dim1)
        self.lin2 = nn.Linear(last_hidden_dim, output_dim2)

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
        x1 = self.lin1(x)
        x2 = self.lin2(x)

        return x1, x2
