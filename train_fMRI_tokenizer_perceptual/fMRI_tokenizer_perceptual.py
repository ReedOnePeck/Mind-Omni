import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict, dataclass, field
from typing import List, Optional
import os, json
from train_decoder_for_perception.fMRI_recons_perceptual import fMRI_recons_perceptron
import inspect



# --- ModelArgs 配置类 ---
@dataclass
class ModelArgs:
    n_voxel: int = 16127  # fMRI data original length
    codebook_size: int = 1024  # Recommended initial value: 1024 - 4096
    codebook_embed_dim: int = 256  # Recommended initial value: 256 or 512
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    entropy_loss_ratio: float = 0.0

    desired_token_num: int = 64

    # For 1D convolutions, ch_mult will determine the number of features after each block
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [16, 8, 4, 2, 1])

    # Base channels for the first convolution in encoder/decoder
    base_channels: int = 64

    # Latent dimension (after encoder, before quantization)
    z_channels: int = 256
    num_res_blocks: int = 2
    dropout_p: float = 0.0

    # Loss weights
    lambda_mse: float = 1.0
    lambda_commitment: float = 0.25  # Renamed from beta for clarity in total loss calculation
    lambda_contrastive: float = 1.0
    lambda_distillation: float = 0.5
    lambda_fine_grained: float = 1.0
    lambda_txt_perceptual_loss: float = 0.5
    lambda_img_perceptual_loss: float = 0.5


    # Masked language modeling parameters
    mask_ratio: float = 0.30  # Masking rate for text tokens
    mlm_temp: float = 1.0  # Temperature for masked language modeling softmax
    clip_sos_token_id: int = 49406  # Placeholder, should be set to the actual [SOS] token ID of your CLIP tokenizer


# --- 关键修改：创建一个被 @torch.compile(disable=True) 标记的辅助函数 ---
@torch.compile(disable=True)
def calculate_fine_grained_loss(mlm_head, predictions_for_masked, mlm_temp, text_input_ids, masked_indices):
    """
    这个函数将被排除在 torch.compile 的优化之外，以 eager 模式运行，
    从而避免编译器在 cross_entropy 上产生 bug。
    """
    logits = mlm_head(predictions_for_masked / mlm_temp)
    labels_for_masked = text_input_ids[masked_indices]

    # 在 eager 模式下，这个 cross_entropy 调用将使用标准、稳定的实现
    fine_grained_loss = F.cross_entropy(logits.to(torch.float32), labels_for_masked)
    return fine_grained_loss

def calculate_batch_pcc(fmri_data: torch.Tensor, recon_fmri: torch.Tensor) -> torch.Tensor:
    """
    计算一个批次中，原始fMRI数据和重建fMRI数据之间的皮尔逊相关系数(PCC)的平均值。

    Args:
        fmri_data (torch.Tensor): 原始fMRI数据批次，形状 (batch_size, n_voxel)。
                                  预期在GPU上。
        recon_fmri (torch.Tensor): 重建fMRI数据批次，形状 (batch_size, n_voxel)。
                                   预期在GPU上。

    Returns:
        torch.Tensor: 该批次中所有样本PCC的平均值 (一个标量)。
                      返回的Tensor也在GPU上。
    """
    # 确保输入是PyTorch Tensor并且在同一设备上
    assert isinstance(fmri_data, torch.Tensor), "fmri_data 必须是 torch.Tensor"
    assert isinstance(recon_fmri, torch.Tensor), "recon_fmri 必须是 torch.Tensor"
    assert fmri_data.device == recon_fmri.device, "fmri_data 和 recon_fmri 必须在同一设备上"
    assert fmri_data.shape == recon_fmri.shape, "fmri_data 和 recon_fmri 的形状必须一致"

    batch_size = fmri_data.shape[0]
    pcc_scores = []

    # 遍历批次中的每一个样本
    for i in range(batch_size):
        # 提取单个样本 (都是一维向量)
        original_signal = fmri_data[i]
        reconstructed_signal = recon_fmri[i]

        # 计算均值
        mean_orig = torch.mean(original_signal)
        mean_recon = torch.mean(reconstructed_signal)

        # 计算中心化后的向量 (x - x_mean)
        centered_orig = original_signal - mean_orig
        centered_recon = reconstructed_signal - mean_recon

        # 计算协方差 (numerator)
        covariance = torch.sum(centered_orig * centered_recon)

        # 计算各自的标准差的乘积 (denominator)
        # 注意：这里torch.sum(centered_orig ** 2)是方差，其平方根是标准差
        bessel_correction_orig = torch.sqrt(torch.sum(centered_orig ** 2))
        bessel_correction_recon = torch.sqrt(torch.sum(centered_recon ** 2))
        denominator = bessel_correction_orig * bessel_correction_recon

        # 计算PCC
        # 添加一个很小的eps防止除以零
        pcc = covariance / (denominator + 1e-8)
        pcc_scores.append(pcc)

    # 将列表转换为张量并计算批次的平均PCC
    # torch.stack 会在新的维度上堆叠张量，保持其在GPU上
    batch_pcc = torch.stack(pcc_scores)
    mean_pcc = torch.mean(batch_pcc)

    return mean_pcc



_fMRI_perceptron_global = fMRI_recons_perceptron(
        input_dim=16127,
        output_dim1=1024,
        output_dim2=29 * 1024,
        hidden_dims=[4096, 4096, 4096, 4096]
    )

checkpoint = torch.load('/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/fMRI_perceptron/coarse_and_fine/checkpoint_epoch_40.pth', map_location='cpu')
fMRI_perceptron_state_dict = checkpoint['model_state_dict']

_fMRI_perceptron_global.load_state_dict(fMRI_perceptron_state_dict)
# 确保 fMRI_perceptron 处于评估模式，因为我们冻结其参数
_fMRI_perceptron_global.eval()

# 冻结 fMRI_perceptron 的所有参数
for param in _fMRI_perceptron_global.parameters():
    param.requires_grad = False
print("fMRI_perceptron_global loaded and parameters frozen.") # 添加一个日志，确认操作


# --- Helper Functions and Modules (adapted for 1D) ---

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch', 'layer']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=min(num_groups, in_channels), num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.BatchNorm1d(in_channels)
    elif norm_type == 'layer':
        # LayerNorm for 1D data often normalizes over the feature dimension
        return nn.LayerNorm(in_channels)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = Normalize(self.out_channels, norm_type)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.dropout(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv2(h)
        return self.nin_shortcut(x) + h


class Downsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        # 使用 stride=4 的卷积进行下采样
        # kernel_size=4, stride=4, padding=0 是一个非重叠的窗口
        # kernel_size=5, stride=4, padding=1 是一个有1个元素重叠的窗口，效果可能更好
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=5, stride=4,
                              padding=1) if with_conv else nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 上采样因子调整为 4.0
        x = F.interpolate(x, scale_factor=4.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

# --- 新增辅助模块 ---
class Upsample2x1D(nn.Module):
    """一个简单的 2 倍上采样模块 (1D)"""
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x



class AttnBlock1D(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, l = q.shape
        q = q.permute(0, 2, 1)  # (B, L, C)
        k = k  # (B, C, L)
        v = v.permute(0, 2, 1)  # (B, L, C)

        # Scale dot product attention
        w_ = torch.bmm(q, k)  # (B, L, L)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=-1)

        # Attend to values
        h_ = torch.bmm(w_, v)  # (B, L, C)
        h_ = h_.permute(0, 2, 1).contiguous()  # (B, C, L)

        h_ = self.proj_out(h_)
        return x + h_


# --- Encoder & Decoder (1D Adaptation) ---

class Encoder1D(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.encoder_ch_mult)
        self.num_downsamples = self.num_resolutions - 1 if self.num_resolutions > 0 else 0
        self.total_downsample_factor = 4 ** self.num_downsamples

        self.conv_in = nn.Conv1d(1, config.base_channels, kernel_size=3, stride=1, padding=1)

        # 主体卷积块 (保持您修复后的健壮版本，完全不变)
        self.conv_blocks = nn.ModuleList()
        in_ch = config.base_channels
        for i_level in range(self.num_resolutions):
            block_out_ch = config.base_channels * config.encoder_ch_mult[i_level]
            res_blocks = nn.ModuleList()
            if config.num_res_blocks > 0:
                res_blocks.append(ResnetBlock1D(in_ch, block_out_ch, dropout=config.dropout_p))
                for _ in range(config.num_res_blocks - 1):
                    res_blocks.append(ResnetBlock1D(block_out_ch, block_out_ch, dropout=config.dropout_p))
            else:
                res_blocks.append(nn.Conv1d(in_ch, block_out_ch, 1))

            in_ch_for_downsample = block_out_ch  # 这一行在您的原始代码中缺失，但应该是这样
            downsample = nn.Identity() if i_level == self.num_resolutions - 1 else Downsample1D(in_ch_for_downsample,
                                                                                                with_conv=True)
            self.conv_blocks.append(nn.ModuleDict({'res': res_blocks, 'downsample': downsample}))
            in_ch = block_out_ch

        # --- 【核心修改】在中间层前增加一个可选的 2 倍下采样层 ---
        self.final_downsample_factor = self.total_downsample_factor
        if config.desired_token_num == 32:
            self.mid_downsample = nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
            self.final_downsample_factor *= 2
        elif config.desired_token_num == 64:
            self.mid_downsample = nn.Identity()
        else:
            raise ValueError(f"Unsupported desired_token_num: {config.desired_token_num}. Must be 32 or 64.")
        # --- 修改结束 ---

        self.mid = nn.ModuleList([
            ResnetBlock1D(in_ch, in_ch, dropout=config.dropout_p),
            AttnBlock1D(in_ch),
            ResnetBlock1D(in_ch, in_ch, dropout=config.dropout_p)
        ])
        self.norm_out = Normalize(in_ch)
        self.conv_out = nn.Conv1d(in_ch, config.z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 【核心修改】更新填充逻辑以适应新的总下采样因子 ---
        original_len = x.shape[-1]
        # 使用 self.config.desired_token_num 使得 forward 也变成动态的
        target_len = self.config.desired_token_num * self.final_downsample_factor

        if original_len > target_len:
            x = x[:, :target_len]
        else:
            pad_len = target_len - original_len
            x = F.pad(x, (0, pad_len), 'constant', 0)
        # --- 修改结束 ---

        h = x.unsqueeze(1)
        h = self.conv_in(h)
        for block_dict in self.conv_blocks:
            for res_block in block_dict['res']:
                h = res_block(h)
            h = block_dict['downsample'](h)

        # --- 【核心修改】应用这个新增的下采样层 ---
        h = self.mid_downsample(h)
        # --- 修改结束 ---

        for mid_block in self.mid:
            h = mid_block(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder1D(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.decoder_ch_mult)
        self.original_len = config.n_voxel

        block_in_ch = config.base_channels * config.decoder_ch_mult[0]
        self.conv_in = nn.Conv1d(config.z_channels, block_in_ch, kernel_size=3, stride=1, padding=1)

        self.mid = nn.ModuleList([
            ResnetBlock1D(block_in_ch, block_in_ch, dropout=config.dropout_p),
            AttnBlock1D(block_in_ch),
            ResnetBlock1D(block_in_ch, block_in_ch, dropout=config.dropout_p)
        ])

        # --- 【核心修改】在中间层后增加一个可选的 2 倍上采样层，与 Encoder 对称 ---
        if config.desired_token_num == 32:
            self.post_mid_upsample = Upsample2x1D(block_in_ch, with_conv=True)
        elif config.desired_token_num == 64:
            self.post_mid_upsample = nn.Identity()
        else:
            # 保持与 Encoder 的一致性
            raise ValueError(f"Unsupported desired_token_num: {config.desired_token_num}. Must be 32 or 64.")
        # --- 修改结束 ---

        # 主体上采样块 (保持您修复后的健壮版本，完全不变)
        self.conv_blocks = nn.ModuleList()
        in_ch = block_in_ch
        for i_level in range(self.num_resolutions):
            block_out_ch = config.base_channels * config.decoder_ch_mult[i_level]
            res_blocks = nn.ModuleList()
            if config.num_res_blocks + 1 > 0:
                res_blocks.append(ResnetBlock1D(in_ch, block_out_ch, dropout=config.dropout_p))
                for _ in range(config.num_res_blocks):
                    res_blocks.append(ResnetBlock1D(block_out_ch, block_out_ch, dropout=config.dropout_p))
            else:
                res_blocks.append(nn.Conv1d(in_ch, block_out_ch, 1))

            in_ch_for_upsample = block_out_ch  # 这一行在您的原始代码中缺失，但应该是这样
            upsample = nn.Identity() if i_level == self.num_resolutions - 1 else Upsample1D(in_ch_for_upsample,
                                                                                            with_conv=True)
            self.conv_blocks.append(nn.ModuleDict({'res': res_blocks, 'upsample': upsample}))
            in_ch = block_out_ch

        self.norm_out = Normalize(in_ch)
        self.conv_out = nn.Conv1d(in_ch, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        for mid_block in self.mid:
            h = mid_block(h)

        # --- 【核心修改】应用这个新增的上采样层 ---
        h = self.post_mid_upsample(h)
        # --- 修改结束 ---

        for block_dict in self.conv_blocks:
            for res_block in block_dict['res']:
                h = res_block(h)
            h = block_dict['upsample'](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = h.squeeze(1)
        return h[:, :self.original_len]


# --- Vector Quantizer (with EMA) ---
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_e, e_dim, l2_norm, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.l2_norm = l2_norm
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)

        self.register_buffer('ema_cluster_size', torch.zeros(n_e))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(self, z_permuted):
        """
        输入 z_permuted 的形状应为 (B, L, C)
        """
        # --- BUG 1 修复 ---
        # 旧代码: z_flattened = z.permute(0, 2, 1).contiguous().view(-1, self.e_dim)
        # 新代码: 直接使用 z_permuted
        assert self.e_dim == z_permuted.shape[-1]  # 判断输入 z_permuted 的形状是否为 (B, L, C)
        z_flattened = z_permuted.contiguous().view(-1, self.e_dim)

        embedding = F.normalize(self.embedding.weight, p=2, dim=-1) if self.l2_norm else self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, embedding.t())

        min_encoding_indices = torch.argmin(d, dim=1)

        # 恢复到 (B, C, L) 维度进行后续计算
        z = z_permuted.permute(0, 2, 1).contiguous()  # Encoder输出的结果
        z_q_permuted = self.embedding(min_encoding_indices).view(z_permuted.shape)
        z_q = z_q_permuted.permute(0, 2, 1).contiguous()

        # 在 (B, C, L) 维度上计算 loss 和 STE
        commit_loss = F.mse_loss(z_q.detach(), z)
        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # EMA update
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(min_encoding_indices, self.n_e).float()
                # 更新 EMA 计数和加权和
                self.ema_cluster_size.data.mul_(self.decay).add_(torch.sum(encodings, 0), alpha=1 - self.decay)
                self.ema_w.data.mul_(self.decay).add_(torch.matmul(encodings.t(), z_flattened), alpha=1 - self.decay)

                # --- 拉普拉斯平滑 (保持不变，第一道防线) ---
                n = torch.sum(self.ema_cluster_size.data)
                smoothed_cluster_size = (self.ema_cluster_size + self.eps) / (n + self.n_e * self.eps) * n

                # --- 增加数值稳定性检查 (第二道防线) ---
                # 计算新的码本向量 (只在内存中，不直接赋值)
                new_embeddings = self.ema_w / smoothed_cluster_size.unsqueeze(1)

                # 创建一个掩码，标记哪些码本向量是“活跃”的
                # 如果一个向量的有效计数大于1 (可以设为其他小阈值), 我们才认为它是活跃的
                active_mask = (self.ema_cluster_size > 1.0).unsqueeze(1)

                # 获取当前的 embedding 权重
                current_embeddings = self.embedding.weight.data

                # 使用掩码来组合新旧 embedding
                # 只更新活跃的向量，不活跃的向量保持原样
                updated_embeddings = torch.where(
                    active_mask,
                    new_embeddings,
                    current_embeddings
                )

                # 将最终安全更新后的权重复制回去
                self.embedding.weight.data.copy_(updated_embeddings)

                if self.l2_norm:
                    self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)

        codebook_usage = len(torch.unique(min_encoding_indices)) / self.n_e

        # 返回 (B, C, L) 形状的 z_q, loss, 和 (B*L) 的 indices
        return z_q, commit_loss, min_encoding_indices, codebook_usage


# --- Multi-Modal Modules ---
class CLIPProjector(nn.Module):
    def __init__(self, codebook_embed_dim: int, clip_embed_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(codebook_embed_dim, clip_embed_dim * 2),
            nn.GELU(),
            nn.Linear(clip_embed_dim * 2, clip_embed_dim)
        )

    def forward(self, fmri_tokens_avg):
        return self.mlp(fmri_tokens_avg)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, query, context, mask=None):
        b, n, _, h = *query.shape, self.num_heads
        q = self.to_q(query)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            mask = mask.view(b, 1, 1, -1).expand_as(sim)
            sim = sim.masked_fill(mask == 0, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class MaskedLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return self.decoder(self.ln(self.gelu(self.dense(x))))


class VQ_fMRI(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        # Encoder, Decoder, 和 Quantization 部分与之前相同，无需改动
        self.encoder = Encoder1D(config)
        self.decoder = Decoder1D(config)
        self.quant_conv = nn.Conv1d(config.z_channels, config.codebook_embed_dim, 1)
        self.quantize = VectorQuantizerEMA(
            config.codebook_size, config.codebook_embed_dim, config.codebook_l2_norm
        )
        self.post_quant_conv = nn.Conv1d(config.codebook_embed_dim, config.z_channels, 1)

        # 多模态组件
        self.desired_token_num = config.desired_token_num
        concatenated_dim = self.desired_token_num * config.codebook_embed_dim
        self.fmri_to_clip_proj = CLIPProjector(concatenated_dim, clip_embed_dim=1024)

        # 交叉注意力模块 (Query: Text, Context: fMRI)
        self.cross_attention = CrossAttention(
            query_dim=1024, context_dim=config.codebook_embed_dim
        )
        # MLM预测头
        self.mlm_head = MaskedLMHead(hidden_size=1024, vocab_size=49408)

        # 新增: 可学习的 <mask> token embedding，用于替换被掩码的文本特征
        # 维度与CLIP hidden feature一致 (1024)
        self.masked_text_embedding = nn.Parameter(torch.randn(1, 1, 1024))

        # 初始化可学习的CLIP温度系数
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))


        self.fMRI_perceptron = _fMRI_perceptron_global
        # 由于 _fMRI_perceptron_global 已经在文件顶层被冻结，这里不需要再次冻结
        # 但是为了代码的健壮性，我们可以加一个断言来确认
        assert not any(p.requires_grad for p in self.fMRI_perceptron.parameters()), \
            "Error: fMRI_perceptron parameters are not frozen!"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        """
        从给定的路径加载预训练模型权重和配置。

        【已更新】: 此版本会智能过滤掉 config.json 中存在但 ModelArgs 类中已不存在的参数，
                   避免因配置版本不匹配导致的 TypeError。

        Args:
            pretrained_model_path (str): 包含 config.json 和 pytorch_model.bin 的文件夹路径。

        Returns:
            VQ_fMRI: 加载了预训练权重的模型实例。
        """
        print(f"Loading model from {pretrained_model_path}")
        config_path = os.path.join(pretrained_model_path, "config.json")
        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # 1. 加载配置文件
        with open(config_path, "r") as f:
            config_dict_from_file = json.load(f)

        # ==================== 【关键修改】 ====================
        # 2. 获取 ModelArgs 类中所有已定义的字段名
        #    对于 dataclass，我们可以直接从 __dataclass_fields__ 获取
        if hasattr(ModelArgs, '__dataclass_fields__'):
            expected_keys = set(ModelArgs.__dataclass_fields__.keys())
        else:
            # 一个备选方案，适用于非 dataclass 的普通类
            expected_keys = set(inspect.signature(ModelArgs).parameters.keys())

        # 3. 过滤加载的配置字典，只保留 ModelArgs 中存在的键
        filtered_config_dict = {
            key: value for key, value in config_dict_from_file.items()
            if key in expected_keys
        }

        # (可选) 打印出被忽略的键，方便调试
        ignored_keys = set(config_dict_from_file.keys()) - expected_keys
        if ignored_keys:
            print(f"Ignoring unexpected keys in config.json: {', '.join(ignored_keys)}")

        # 4. 使用过滤后的字典安全地初始化 ModelArgs
        config = ModelArgs(**filtered_config_dict)
        # =====================================================

        # 5. 使用加载的配置初始化模型
        model = cls(config)

        # 6. 加载预训练权重
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)  # 建议加上 weights_only=True

        # 处理 `torch.compile` 可能添加的 `_orig_mod.` 前缀
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                cleaned_state_dict[k.removeprefix("_orig_mod.")] = v
            else:
                cleaned_state_dict[k] = v

        # 加载权重到模型中
        load_result = model.load_state_dict(cleaned_state_dict, strict=True)
        print(f"Weight loading result: {load_result}")

        # 确保模型处于评估模式
        model.eval()

        return model

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "config.json")
        model_path = os.path.join(output_dir, "pytorch_model.bin")

        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        torch.save(self.state_dict(), model_path)
        print(f"Model and config saved to {output_dir}")


    def forward_for_inference(self, fmri_data: torch.Tensor):
        """
        用于推理(Inference): 将fMRI数据编码并量化为离散的token.

        Args:
            fmri_data (torch.Tensor): 输入的fMRI数据, shape: (B, n_voxel).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - quantized_fmri_tokens (torch.Tensor): 量化后的fMRI token, shape: (B, L, codebook_embed_dim).
                - codebook_indices (torch.Tensor): 对应的码本索引, shape: (B, L).
        """
        # 1. 编码
        h = self.encoder(fmri_data)
        h = self.quant_conv(h)  # (B, C, L)

        # 2. 量化
        # VectorQuantizerEMA需要(B, L, C)输入, 所以需要permute
        quantized_fmri, _, indices, _ = self.quantize(h.permute(0, 2, 1))

        # 将输出格式统一为 (B, L, C)
        quantized_fmri_tokens = quantized_fmri.permute(0, 2, 1)
        codebook_indices = indices.view(quantized_fmri_tokens.shape[0], -1)

        return quantized_fmri_tokens, codebook_indices

    def forward(self,
                fmri_data: torch.Tensor,
                img_clip_feature: torch.Tensor,
                text_clip_feature: torch.Tensor,
                text_clip_hidden_features: torch.Tensor,
                text_input_ids: torch.Tensor,
                text_padding_mask: torch.Tensor,
                return_loss_dict: bool = False):
        """
        用于训练(Training): 计算所有相关的损失.
        """
        # --- 1. VQ-GAN 编码、量化与重建 ---
        h = self.encoder(fmri_data)
        h_pre_quant = self.quant_conv(h)

        quantized_fmri_conv, commit_loss, _, codebook_usage = self.quantize(
            h_pre_quant.permute(0, 2, 1))  # quantized_fmri_conv的形状是(B,C,L)

        recon_fmri = self.decoder(self.post_quant_conv(quantized_fmri_conv))
        recon_loss = F.mse_loss(recon_fmri, fmri_data)

        losses = {"recon_loss": recon_loss, "commit_loss": commit_loss}

        # --- 2. 粗粒度损失 ---
        fmri_tokens = quantized_fmri_conv.permute(0, 2, 1)
        batch_size = fmri_tokens.shape[0]
        fmri_tokens_flattened = fmri_tokens.reshape(batch_size, -1)  # Shape: (B, 16 * 64) -> (B, 1024)
        fmri_clip_projected = self.fmri_to_clip_proj(fmri_tokens_flattened)

        fmri_norm = F.normalize(fmri_clip_projected, p=2, dim=-1)
        img_norm = F.normalize(img_clip_feature, p=2, dim=-1)
        text_norm = F.normalize(text_clip_feature, p=2, dim=-1)

        # 获取学习到的温度系数
        logit_scale = self.logit_scale.exp()
        logits_fmri_img = torch.matmul(fmri_norm, img_norm.t()) * logit_scale
        logits_fmri_text = torch.matmul(fmri_norm, text_norm.t()) * logit_scale

        labels = torch.arange(fmri_data.shape[0], device=fmri_data.device)

        clip_loss_img = (F.cross_entropy(logits_fmri_img, labels) + F.cross_entropy(logits_fmri_img.t(), labels)) / 2
        clip_loss_text = (F.cross_entropy(logits_fmri_text, labels) + F.cross_entropy(logits_fmri_text.t(), labels)) / 2

        losses["contrastive_loss"] = clip_loss_img + clip_loss_text
        losses["distillation_loss"] = F.mse_loss(fmri_clip_projected, img_clip_feature.detach())

        # --- 3. 细粒度损失 (Fine-grained Loss: Masked Language Modeling) - 已采纳建议修改 ---
        batch_size = fmri_data.shape[0]
        can_be_masked_mask = (text_padding_mask == 1) & (text_input_ids != self.config.clip_sos_token_id)

        masked_indices = torch.zeros_like(text_input_ids, dtype=torch.bool)
        for i in range(batch_size):
            valid_indices = torch.where(can_be_masked_mask[i])[0]
            if len(valid_indices) > 0:
                num_to_mask = max(1, int(len(valid_indices) * self.config.mask_ratio))
                perm = torch.randperm(len(valid_indices), device=fmri_data.device)
                indices_to_mask = valid_indices[perm[:num_to_mask]]
                masked_indices[i, indices_to_mask] = True

        query_input = text_clip_hidden_features.clone()

        # --- 关键修复 ---
        # 在赋值之前，将 self.masked_text_embedding 转换为与 query_input 相同的 dtype
        # 这样可以保证类型匹配
        masked_embedding_casted = self.masked_text_embedding.to(query_input.dtype)

        # 然后用可学习的<mask> embedding替换掉需要被掩码的位置
        # PyTorch的广播机制会自动处理形状匹配
        query_input[masked_indices] = masked_embedding_casted

        # Text(被掩码过的Query) attends to fMRI(Context)
        text_aligned_to_fmri = self.cross_attention(
            query=query_input,
            context=fmri_tokens
        )

        predictions_for_masked = text_aligned_to_fmri[masked_indices]

        # --- 关键修改：调用我们新的辅助函数 ---
        fine_grained_loss = calculate_fine_grained_loss(
            self.mlm_head,
            predictions_for_masked,
            self.config.mlm_temp,
            text_input_ids,
            masked_indices
        )

        losses["fine_grained_loss"] = fine_grained_loss

        # --- 4. 计算perceptual 损失---
        pred_img, pred_txt = self.fMRI_perceptron(recon_fmri)
        img_perceptual_loss = F.mse_loss(pred_img, img_clip_feature)
        text_hidden_gt_no_sos = text_clip_hidden_features[:,1:,:]
        txt_perceptual_loss =  F.mse_loss(pred_txt, text_hidden_gt_no_sos.reshape(text_hidden_gt_no_sos.shape[0], -1))

        losses["img_perceptual_loss"] = img_perceptual_loss
        losses["txt_perceptual_loss"] = txt_perceptual_loss

        # --- 5. 计算总损失 ---
        total_loss = (self.config.lambda_mse * losses["recon_loss"] +
                      self.config.lambda_commitment * losses["commit_loss"] +
                      self.config.lambda_contrastive * losses["contrastive_loss"] +
                      self.config.lambda_distillation * losses["distillation_loss"] +
                      self.config.lambda_fine_grained * losses["fine_grained_loss"] +
                      self.config.lambda_img_perceptual_loss * losses["img_perceptual_loss"] +
                      self.config.lambda_txt_perceptual_loss * losses["txt_perceptual_loss"]
                      )

        losses["total_loss"] = total_loss
        losses["codebook_usage"] = codebook_usage

        # ----------6.计算PCC----------------
        recons_pcc = calculate_batch_pcc(fmri_data, recon_fmri)
        losses["recons_pcc"] = recons_pcc

        perceptual_img_pcc = calculate_batch_pcc(pred_img, img_clip_feature)
        perceptual_txt_pcc = calculate_batch_pcc(pred_txt, text_hidden_gt_no_sos.reshape(text_hidden_gt_no_sos.shape[0], -1))

        losses["perceptual_img_pcc"] = perceptual_img_pcc
        losses["perceptual_txt_pcc"] = perceptual_txt_pcc

        if return_loss_dict:
            # 如果需要完整的日志，返回整个字典
            return losses
        else:
            # 默认情况下，只返回用于反向传播的总损失
            # 这让 DDP 的追踪路径变得极其简单清晰
            return total_loss

    def calculate_pcc(self, fmri_data: torch.Tensor) -> torch.Tensor:
        """
        计算一个批次中，原始fMRI和重建fMRI之间的皮尔逊相关系数(PCC).

        Args:
            fmri_data (torch.Tensor): 输入的原始fMRI数据批次,
                                      shape: (B, n_voxel).

        Returns:
            torch.Tensor: 该批次中所有样本PCC的平均值 (一个标量).


        # ... 在您的评估循环中 ...
        all_pcc_scores = []
        for batch in test_loader:
            # 假设batch是一个字典或元组，其中包含了fmri_data
            fmri_data_batch = batch['fmri'].to(device) # 将数据移动到GPU

            # 调用函数计算该批次的平均PCC
            mean_pcc_for_batch = fMRI_quantizer.calculate_pcc(fmri_data_batch)

            all_pcc_scores.append(mean_pcc_for_batch.item()) # .item() 将GPU标量转为Python数字

        # 计算整个测试集的平均PCC
        average_pcc_over_dataset = sum(all_pcc_scores) / len(all_pcc_scores)
        print(f"整个测试集的平均PCC为: {average_pcc_over_dataset:.4f}")
        """
        # 确保模型处于评估模式，这会关闭dropout等层
        self.eval()

        # 使用 torch.no_grad() 来禁用梯度计算，节省显存和计算资源
        with torch.no_grad():
            # --- 1. 前向传播，获取重建的fMRI ---
            h = self.encoder(fmri_data)
            h_pre_quant = self.quant_conv(h)

            # self.quantize 接收 (B, L, C) 并返回 (B, C, L)
            quantized_fmri_conv, _, _, _ = self.quantize(h_pre_quant.permute(0, 2, 1))

            # 直接使用 (B, C, L) 格式的输出
            recon_fmri = self.decoder(self.post_quant_conv(quantized_fmri_conv))

            # --- 2. 计算皮尔逊相关系数 ---
            batch_size = fmri_data.shape[0]
            pcc_scores = []

            # 遍历批次中的每一个样本
            for i in range(batch_size):
                # 提取单个样本 (都是一维向量)
                original_signal = fmri_data[i]
                reconstructed_signal = recon_fmri[i]

                # 计算均值
                mean_orig = torch.mean(original_signal)
                mean_recon = torch.mean(reconstructed_signal)

                # 计算中心化后的向量 (x - x_mean)
                centered_orig = original_signal - mean_orig
                centered_recon = reconstructed_signal - mean_recon

                # 计算协方差 (numerator)
                covariance = torch.sum(centered_orig * centered_recon)

                # 计算各自的标准差的乘积 (denominator)
                bessel_correction_orig = torch.sqrt(torch.sum(centered_orig ** 2))
                bessel_correction_recon = torch.sqrt(torch.sum(centered_recon ** 2))
                denominator = bessel_correction_orig * bessel_correction_recon

                # 计算PCC
                # 添加一个很小的eps防止除以零
                pcc = covariance / (denominator + 1e-8)
                pcc_scores.append(pcc)

            # 将列表转换为张量并计算批次的平均PCC
            batch_pcc = torch.stack(pcc_scores)
            mean_pcc = torch.mean(batch_pcc)

        return mean_pcc

    def calculate_retrieval(self,
                            fmri_data: torch.Tensor,
                            img_clip_feature: torch.Tensor,
                            text_clip_feature: torch.Tensor) -> dict:
        """
        Args:
            fmri_data (torch.Tensor): 整个测试集的fMRI数据, shape: (N, n_voxel), e.g., (1000, 16127).
            img_clip_feature (torch.Tensor): 对应的图像CLIP特征, shape: (N, clip_dim), e.g., (1000, 1024).
            text_clip_feature (torch.Tensor): 对应的文本CLIP特征, shape: (N, clip_dim), e.g., (1000, 1024).

        Returns:
            dict: 一个包含检索结果的字典,
                  e.g., {'fmri_to_image_acc': 0.85, 'fmri_to_text_acc': 0.82}.


        #测试样例
        retrieval_results = fMRI_quantizer.calculate_retrieval(
        test_fmri_data,
        test_img_features,
        test_text_features)

        print("--- 跨模态检索评估结果 ---")
        print(f"fMRI -> 图像 检索准确率: {retrieval_results['fmri_to_image_acc']:.4f}")
        print(f"fMRI -> 文本 检索准确率: {retrieval_results['fmri_to_text_acc']:.4f}")
        print(f"随机准确率 (Chance Level): {1/len(test_fmri_data):.4f}")
        """

        self.eval()
        with torch.no_grad():
            # --- 1. 将所有fMRI数据转换为CLIP空间的特征 ---
            h = self.encoder(fmri_data)
            h_pre_quant = self.quant_conv(h)
            quantized_fmri_conv, _, _, _ = self.quantize(h_pre_quant.permute(0, 2, 1))

            fmri_tokens = quantized_fmri_conv.permute(0, 2, 1)
            batch_size = fmri_tokens.shape[0]
            fmri_tokens_flattened = fmri_tokens.reshape(batch_size, -1)  # Shape: (B, 16 * 64) -> (B, 1024)
            fmri_clip_projected = self.fmri_to_clip_proj(fmri_tokens_flattened)
            """
            fmri_tokens = quantized_fmri_conv.permute(0, 2, 1)
            fmri_tokens_avg = fmri_tokens.mean(dim=1)
            fmri_clip_projected = self.fmri_to_clip_proj(fmri_tokens_avg)
            """

            # --- 2. 特征归一化 (为计算余弦相似度做准备) ---
            fmri_norm = F.normalize(fmri_clip_projected, p=2, dim=-1)
            img_norm = F.normalize(img_clip_feature, p=2, dim=-1)
            text_norm = F.normalize(text_clip_feature, p=2, dim=-1)

            # --- 3. 计算fMRI到图像的检索 (fMRI-to-Image Retrieval) ---

            # 计算相似度矩阵. (N_fmri, D) x (D, N_img) -> (N_fmri, N_img)
            # 结果 sim_matrix_img[i, j] 表示第i个fMRI和第j个图像的余弦相似度
            sim_matrix_img = torch.matmul(fmri_norm, img_norm.t())

            # 找到每一行中最大值的索引.
            # predicted_indices_img[i] 是模型认为与第i个fMRI最匹配的图像的索引
            predicted_indices_img = torch.argmax(sim_matrix_img, dim=1)

            # --- 4. 计算fMRI到文本的检索 (fMRI-to-Text Retrieval) ---

            # 计算相似度矩阵. (N_fmri, D) x (D, N_text) -> (N_fmri, N_text)
            sim_matrix_text = torch.matmul(fmri_norm, text_norm.t())

            # 找到每一行中最大值的索引
            predicted_indices_text = torch.argmax(sim_matrix_text, dim=1)

            # --- 5. 统计准确率 ---

            # 创建正确答案的标签. 因为数据是对齐的, 正确的索引就是 [0, 1, 2, ..., N-1]
            num_samples = fmri_data.shape[0]
            ground_truth_indices = torch.arange(num_samples, device=fmri_data.device)

            # 比较预测和真实标签，计算正确的数量
            correct_img_retrievals = torch.sum(predicted_indices_img == ground_truth_indices)
            correct_text_retrievals = torch.sum(predicted_indices_text == ground_truth_indices)

            # 计算准确率
            fmri_to_image_acc = correct_img_retrievals.float() / num_samples
            fmri_to_text_acc = correct_text_retrievals.float() / num_samples

            # 构建返回结果的字典
            results = {
                'fmri_to_image_acc': fmri_to_image_acc,  # .item() 将GPU标量转为Python数字
                'fmri_to_text_acc': fmri_to_text_acc
            }

        return results


if __name__ == '__main__':
    # 1. 创建配置实例
    config = ModelArgs()

    # 2. 实例化编码器和解码器
    encoder = Encoder1D(config)
    decoder = Decoder1D(config)
    fMRI_quantizer = VQ_fMRI(config)

    brain_vocab = fMRI_quantizer.quantize.embedding.weight.data



    # 3. 模拟一次前向传播来验证维度
    #    输入fmri数据 (Batch size = 4)
    fmri_data = torch.randn(4, config.n_voxel)
    print(f"输入 fMRI 形状: {fmri_data.shape}")


    img_clip_feature = torch.randn(4, 1024)
    text_clip_feature = torch.randn(4, 1024)

    DATA_DIR = "/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_features/CLIP_H_text_max30/"

    list_input_ids = []
    list_attention_mask = []
    list_last_hidden_state = []

    num_files_to_load = 4
    print(f"Attempting to load the first {num_files_to_load} files from '{DATA_DIR}'...")

    # --- 3. 循环读取文件并收集数据 ---
    for i in range(num_files_to_load):
        # 构建完整的文件路径，例如 '.../00000.pt', '.../00001.pt', ...
        file_name = f"{i:05d}.pt"
        file_path = os.path.join(DATA_DIR, file_name)

        try:
            # 加载.pt文件，它包含一个字典
            data_dict = torch.load(file_path)

            # 从字典中提取三个张量，并添加到相应的列表中
            list_input_ids.append(data_dict['input_ids'])
            list_attention_mask.append(data_dict['attention_mask'])
            list_last_hidden_state.append(data_dict['hidden_state'])

            print(f"Successfully loaded and parsed {file_name}")

        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    text_input_ids = torch.stack(list_input_ids, dim=0)
    text_padding_mask = torch.stack(list_attention_mask, dim=0)
    text_clip_hidden_features = torch.stack(list_last_hidden_state, dim=0)


    losses = fMRI_quantizer(fmri_data, img_clip_feature, text_clip_feature,text_clip_hidden_features, text_input_ids, text_padding_mask, True)


    print('')


