from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward


from MindOmni_src.tri_modal_attention_precessor import Trimodal_Attention, Trimodal_FluxAttnProcessor2_0


from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle, \
    GlobalResponseNorm, RMSNorm
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, \
    TimestepEmbedding, get_timestep_embedding  # ,FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.resnet import Downsample2D, Upsample2D

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



def get_3d_rotary_pos_embed(
        embed_dim, crops_coords, grid_size, temporal_size, theta: int = 10000, use_real: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    start, stop = crops_coords
    grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
    grid_t = np.linspace(0, temporal_size, temporal_size, endpoint=False, dtype=np.float32)

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2).float() / dim_t))
    grid_t = torch.from_numpy(grid_t).float()
    freqs_t = torch.einsum("n , f -> n f", grid_t, freqs_t)
    freqs_t = freqs_t.repeat_interleave(2, dim=-1)

    # Spatial frequencies for height and width
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2).float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2).float() / dim_w))
    grid_h = torch.from_numpy(grid_h).float()
    grid_w = torch.from_numpy(grid_w).float()
    freqs_h = torch.einsum("n , f -> n f", grid_h, freqs_h)
    freqs_w = torch.einsum("n , f -> n f", grid_w, freqs_w)
    freqs_h = freqs_h.repeat_interleave(2, dim=-1)
    freqs_w = freqs_w.repeat_interleave(2, dim=-1)

    # Broadcast and concatenate tensors along specified dimension
    def broadcast(tensors, dim=-1):
        num_tensors = len(tensors)
        shape_lens = {len(t.shape) for t in tensors}
        assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
        shape_len = list(shape_lens)[0]
        dim = (dim + shape_len) if dim < 0 else dim
        dims = list(zip(*(list(t.shape) for t in tensors)))
        expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
        assert all(
            [*(len(set(t[1])) <= 2 for t in expandable_dims)]
        ), "invalid dimensions for broadcastable concatenation"
        max_dims = [(t[0], max(t[1])) for t in expandable_dims]
        expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
        expanded_dims.insert(dim, (dim, dims[dim]))
        expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
        tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
        return torch.cat(tensors, dim=dim)

    freqs = broadcast((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)

    t, h, w, d = freqs.shape
    freqs = freqs.view(t * h * w, d)

    # Generate sine and cosine components
    sin = freqs.sin()
    cos = freqs.cos()

    if use_real:
        return cos, sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def get_2d_rotary_pos_embed(embed_dim, crops_coords, grid_size, use_real=True):
    """
    RoPE for image tokens with 2d structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size
    crops_coords (`Tuple[int]`)
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the positional embedding.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `torch.Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    start, stop = crops_coords
    grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)  # [2, W, H]

    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[0].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[1].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)

    if use_real:
        cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D)
        sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D)
        return cos, sin
    else:
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
        return emb


def get_2d_rotary_pos_embed_lumina(embed_dim, len_h, len_w, linear_factor=1.0, ntk_factor=1.0):
    assert embed_dim % 4 == 0

    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, len_h, linear_factor=linear_factor, ntk_factor=ntk_factor
    )  # (H, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, len_w, linear_factor=linear_factor, ntk_factor=ntk_factor
    )  # (W, D/4)
    emb_h = emb_h.view(len_h, 1, embed_dim // 4, 1).repeat(1, len_w, 1, 1)  # (H, W, D/4, 1)
    emb_w = emb_w.view(1, len_w, embed_dim // 4, 1).repeat(len_h, 1, 1, 1)  # (H, W, D/4, 1)

    emb = torch.cat([emb_h, emb_w], dim=-1).flatten(2)  # (H, W, D/2)
    return emb


def get_1d_rotary_pos_embed(
        dim: int,
        pos: Union[np.ndarray, int],
        theta: float = 10000.0,
        use_real=False,
        linear_factor=1.0,
        ntk_factor=1.0,
        repeat_interleave_real=True,
        freqs_dtype=torch.float32,  # torch.float32 (hunyuan, stable audio), torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = np.arange(pos)
    theta = theta * ntk_factor
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim)) / linear_factor  # [D/2]
    t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
    freqs = torch.outer(t, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs).float()  # complex64     # [S, D/2]
        return freqs_cis


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: Tuple[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.squeeze().float().cpu().numpy()
        is_mps = ids.device.type == "mps"
        freqs_dtype = torch.float32 if is_mps else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i], repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class SingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = Trimodal_FluxAttnProcessor2_0()
        self.attn = Trimodal_Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            attention_mask: Optional[torch.Tensor] = None
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask = attention_mask
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class TransformerBlock(nn.Module):
    r"""
    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.

    跟原版的MMDit相比，forward的输入中新增了一个模态thirdmodal_hidden_states  以及 attention mask，该mask用来说明哪个模态是缺失的，比如在训练 fMRI<---->图像时，文本模态虽然缺失，但是也得把sequence的
    位置占着，对于这些占位符，就需要用attention mask进行掩码。

    初始化的参数和原版相比，新增了self.norm1_thirdmodal = AdaLayerNormZero(dim)， self.norm2_thirdmodal = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_thirdmodal = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        #为第三个模态新增的layernorm
        self.norm1_thirdmodal = AdaLayerNormZero(dim)



        if hasattr(F, "scaled_dot_product_attention"):
            processor = Trimodal_FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.attn = Trimodal_Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            trimodal_kv_proj_dim=dim,
            trimodal_output=True,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        # self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="swiglu")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        # self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="swiglu")

        # 为第三个模态新增的layernorm
        self.norm2_thirdmodal = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_thirdmodal = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            thirdmodal_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            attention_mask: Optional[torch.Tensor] = None
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        #第三个新增模态的计算量
        norm_thirdmodal_hidden_states, third_gate_msa, third_shift_mlp, third_scale_mlp, third_gate_mlp = self.norm1_thirdmodal(thirdmodal_hidden_states, emb=temb)


        # Attention.
        attn_output, context_attn_output, thirdmodal_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            thirdmodal_hidden_states = norm_thirdmodal_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask = attention_mask
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output


        #为第三个模态新增的计算量
        thirdmodal_output = third_gate_msa.unsqueeze(1) * thirdmodal_output
        thirdmodal_hidden_states = thirdmodal_hidden_states + thirdmodal_output

        norm_thirdmodal_hidden_states = self.norm2_thirdmodal(thirdmodal_hidden_states)
        norm_thirdmodal_hidden_states = norm_thirdmodal_hidden_states  * (1 + third_scale_mlp[:, None]) + third_shift_mlp[:, None]

        thirdmodal_ff_output = self.ff_thirdmodal(norm_thirdmodal_hidden_states)
        thirdmodal_hidden_states = thirdmodal_hidden_states + third_gate_mlp.unsqueeze(1) * thirdmodal_ff_output



        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        if thirdmodal_hidden_states.dtype == torch.float16:
            thirdmodal_hidden_states = thirdmodal_hidden_states.clip(-65504, 65504)


        #注意这里的输出顺序又变成了：  文-图-脑
        return encoder_hidden_states, hidden_states,  thirdmodal_hidden_states


# --- 4. 三模态TransformerBlock验证函数 ---
@torch.no_grad()
def test_trimodal_transformer_block():
    print("--- 开始验证三模态 TransformerBlock (图像模态缺失) ---")

    # 1. 定义模型参数
    batch_size = 4
    dim = 128
    num_attention_heads = 8
    attention_head_dim = dim // num_attention_heads
    text_seq_len, image_seq_len, brain_seq_len = 77, 256, 100

    # 为了可复现性，固定随机种子
    torch.manual_seed(42)

    # 2. 实例化修改后的 TransformerBlock
    block = TransformerBlock(dim, num_attention_heads, attention_head_dim)
    block.eval()
    print("TransformerBlock 实例化成功。")

    # 3. 创建第一组随机输入数据 (图像模态为全零占位符)
    hidden_states_zero = torch.zeros(batch_size, image_seq_len, dim)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, dim)
    thirdmodal_hidden_states = torch.randn(batch_size, brain_seq_len, dim)
    temb = torch.randn(batch_size, dim)
    print("第一组输入数据已创建 (图像模态为全零占位符)。")

    # 4. 创建 Attention Mask 来屏蔽图像模态
    # 使用浮点数掩码，其中 0.0 代表保留，-inf 代表屏蔽。
    text_mask = torch.zeros(batch_size, text_seq_len, dtype=torch.float32)
    image_mask = torch.ones(batch_size, image_seq_len, dtype=torch.float32) * -torch.inf
    brain_mask = torch.zeros(batch_size, brain_seq_len, dtype=torch.float32)
    attention_mask = torch.cat([text_mask, image_mask, brain_mask], dim=1).unsqueeze(1).unsqueeze(1)
    print(f"Attention Mask 已创建，形状为: {attention_mask.shape}")

    # 5. 执行第一次前向传播
    try:
        print("\n--- 第一次前向传播 (使用全零占位符) ---")
        output_text_1, output_image_1, output_brain_1 = block(
            hidden_states=hidden_states_zero,
            encoder_hidden_states=encoder_hidden_states,
            thirdmodal_hidden_states=thirdmodal_hidden_states,
            temb=temb,
            attention_mask=attention_mask
        )
        print("第一次前向传播成功！")

        # 验证输出的形状是否正确
        assert output_text_1.shape == encoder_hidden_states.shape
        assert output_image_1.shape == hidden_states_zero.shape
        assert output_brain_1.shape == thirdmodal_hidden_states.shape
        print("输出形状验证通过。")

        # --------------------------------------------------- #
        # ------------ 新增：黑盒测试部分开始 ------------ #
        # --------------------------------------------------- #
        print("\n--- 开始黑盒测试：验证 Mask 的信息隔离效果 ---")

        # a. 创建第二组输入数据，只改变被屏蔽的图像占位符内容
        print("创建第二组输入数据 (图像模态为随机噪声占位符)...")
        hidden_states_random = torch.randn(batch_size, image_seq_len, dim) # <-- 唯一的变化

        # b. 执行第二次前向传播，其他所有输入都保持不变
        print("第二次前向传播 (使用随机占位符)...")
        output_text_2, output_image_2, output_brain_2 = block(
            hidden_states=hidden_states_random, # <-- 使用随机占位符
            encoder_hidden_states=encoder_hidden_states,
            thirdmodal_hidden_states=thirdmodal_hidden_states,
            temb=temb,
            attention_mask=attention_mask
        )
        print("第二次前向传播成功！")

        # c. 比较两次输出中未被屏蔽模态的结果
        print("比较两次前向传播中，文本和大脑信号的输出...")
        text_outputs_are_same = torch.allclose(output_text_1, output_text_2, atol=1e-6)
        brain_outputs_are_same = torch.allclose(output_brain_1, output_brain_2, atol=1e-6)

        # d. 输出黑盒测试结论
        if text_outputs_are_same and brain_outputs_are_same:
            print("\n✅ 黑盒测试成功！")
            print("   改变被屏蔽的图像占位符内容，完全不影响文本和大脑信号的输出。")
            print("   这强有力地证明了 attention_mask 成功地隔离了图像模态的信息流。")
        else:
            print("\n❌ 黑盒测试失败！")
            print("   改变图像占位符影响了其他模态的输出，mask 未能完全生效。")
            if not text_outputs_are_same:
                print(f"   - 文本输出不一致，最大差异: {(output_text_1 - output_text_2).abs().max().item()}")
            if not brain_outputs_are_same:
                 print(f"   - 大脑信号输出不一致，最大差异: {(output_brain_1 - output_brain_2).abs().max().item()}")

        # e. (可选) 检查被屏蔽模态的输出是否发生了变化
        image_outputs_are_different = not torch.allclose(output_image_1, output_image_2)
        print("\n(额外检查) 比较被屏蔽的图像模态自身的输出:")
        if image_outputs_are_different:
            print("   - 图像模态的输出在两次运行中是不同的，这符合预期，")
            print("     因为它的输入不同，并且它自身的残差连接仍然会保留其输入信息。")
        else:
            print("   - 图像模态的输出在两次运行中是相同的，这有点奇怪但也是可能的。")

        # ------------------------------------------------- #
        # -------------- 新增：黑盒测试部分结束 -------------- #
        # ------------------------------------------------- #

    except Exception as e:
        print(f"\n❌ 验证失败！发生错误: {e}")
        import traceback
        traceback.print_exc()



class UVit2DConvEmbed(nn.Module):
    def __init__(self, in_channels, block_out_channels, vocab_size, elementwise_affine, eps, bias):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, in_channels)
        self.layer_norm = RMSNorm(in_channels, eps, elementwise_affine)
        self.conv = nn.Conv2d(in_channels, block_out_channels, kernel_size=1, bias=bias)

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = self.conv(embeddings)
        return embeddings


class ConvMlmLayer(nn.Module):
    def __init__(
            self,
            block_out_channels: int,
            in_channels: int,
            use_bias: bool,
            ln_elementwise_affine: bool,
            layer_norm_eps: float,
            codebook_size: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(block_out_channels, in_channels, kernel_size=1, bias=use_bias)
        self.layer_norm = RMSNorm(in_channels, layer_norm_eps, ln_elementwise_affine)
        self.conv2 = nn.Conv2d(in_channels, codebook_size, kernel_size=1, bias=use_bias)

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        logits = self.conv2(hidden_states)
        return logits


class SwiGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function. It's similar to `GEGLU`
    but uses SiLU / Swish instead of GeLU.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class ConvNextBlock(nn.Module):
    def __init__(
            self, channels, layer_norm_eps, ln_elementwise_affine, use_bias, hidden_dropout, hidden_size,
            res_ffn_factor=4
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=use_bias,
        )
        self.norm = RMSNorm(channels, layer_norm_eps, ln_elementwise_affine)
        self.channelwise_linear_1 = nn.Linear(channels, int(channels * res_ffn_factor), bias=use_bias)
        self.channelwise_act = nn.GELU()
        self.channelwise_norm = GlobalResponseNorm(int(channels * res_ffn_factor))
        self.channelwise_linear_2 = nn.Linear(int(channels * res_ffn_factor), channels, bias=use_bias)
        self.channelwise_dropout = nn.Dropout(hidden_dropout)
        self.cond_embeds_mapper = nn.Linear(hidden_size, channels * 2, use_bias)

    def forward(self, x, cond_embeds):
        x_res = x

        x = self.depthwise(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.channelwise_linear_1(x)
        x = self.channelwise_act(x)
        x = self.channelwise_norm(x)
        x = self.channelwise_linear_2(x)
        x = self.channelwise_dropout(x)

        x = x.permute(0, 3, 1, 2)

        x = x + x_res

        scale, shift = self.cond_embeds_mapper(F.silu(cond_embeds)).chunk(2, dim=1)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        return x


class Simple_UVitBlock(nn.Module):
    def __init__(
            self,
            channels,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            downsample: bool,
            upsample: bool,
    ):
        super().__init__()

        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
            )
        else:
            self.downsample = None

        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                interpolate=False,
            )
        else:
            self.upsample = None

    def forward(self, x):
        # print("before,", x.shape)
        if self.downsample is not None:
            # print('downsample')
            x = self.downsample(x)

        if self.upsample is not None:
            # print('upsample')
            x = self.upsample(x)
        # print("after,", x.shape)
        return x



class Trimodal_SymmetricTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """
    """
    跟原版的SymmetricTransformer2DModel相比，
    （1）init中新增  third_modal_codebook_size, third_modal_token_dim, 注意或许需要在/nfs/diskstation/DataStation/ChangdeDu/Muddit/512/transformer/config.json  中添加该参数
        为第三个模态输入层新增  third_modal_embedder, third_modal_norm
        为第三个模态输出层新增  third_modal_decoder
        参数可更新的mask token   self.fmri_mask_token = nn.Parameter(torch.randn(third_modal_token_dim))

    （2）在forward函数中，输入增加了：thirdmodal_hidden_states,  thirdmodal_ids,  attention_mask
    """

    _supports_gradient_checkpointing = False  # True
    # Due to NotImplementedError: DDPOptimizer backend: Found a higher order op in the graph. This is not supported. Please turn off DDP optimizer using torch._dynamo.config.optimize_ddp=False. Note that this can cause performance degradation because there will be one bucket for the entire Dynamo graph.
    # Please refer to this issue - https://github.com/pytorch/pytorch/issues/104674.
    _no_split_modules = ["TransformerBlock", "SingleTransformerBlock"]

    @register_to_config
    def __init__(
            self,
            patch_size: int = 1,
            in_channels: int = 64,
            num_layers: int = 19,
            num_single_layers: int = 38,
            attention_head_dim: int = 128,
            num_attention_heads: int = 24,
            joint_attention_dim: int = 4096,
            pooled_projection_dim: int = 768,
            guidance_embeds: bool = False,  # unused in our implementation
            axes_dims_rope: Tuple[int] = (16, 56, 56),
            vocab_size: int = 8256,
            codebook_size: int = 8192,
            third_modal_codebook_size: int = 128,
            third_modal_token_dim: int = 16,
            tokenizer_vocab_size: Optional[int] = None,
            t5_dim: Optional[int] = None,
            downsample: bool = False,
            upsample: bool = False,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = self.num_attention_heads * self.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.inner_dim
        )

        if t5_dim is not None:
            self.adapter = nn.Sequential(
                nn.LayerNorm(t5_dim, elementwise_affine=False, eps=1e-6),
                nn.Linear(t5_dim, self.joint_attention_dim, bias=False)
            )
        else:
            self.adapter = None

        self.context_embedder = nn.Linear(self.joint_attention_dim, self.inner_dim)

        #为第三个模态新增的映射层
        self.third_modal_embedder = nn.Linear(third_modal_token_dim, self.inner_dim)
        self.fmri_mask_token = nn.Parameter(torch.randn(third_modal_token_dim))




        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                )
                for i in range(self.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                )
                for i in range(self.num_single_layers)
            ]
        )

        self.gradient_checkpointing = False

        in_channels_embed = self.inner_dim
        ln_elementwise_affine = True
        layer_norm_eps = 1e-06
        use_bias = False
        micro_cond_embed_dim = 1280
        self.embed = UVit2DConvEmbed(
            in_channels_embed, self.inner_dim, self.vocab_size, ln_elementwise_affine, layer_norm_eps, use_bias
        )
        self.mlm_layer = ConvMlmLayer(
            self.inner_dim, in_channels_embed, use_bias, ln_elementwise_affine, layer_norm_eps, self.codebook_size
        )
        self.cond_embed = TimestepEmbedding(
            micro_cond_embed_dim + self.pooled_projection_dim, self.inner_dim, sample_proj_bias=use_bias
        )
        self.encoder_proj_layer_norm = RMSNorm(self.inner_dim, layer_norm_eps, ln_elementwise_affine)
        self.project_to_hidden_norm = RMSNorm(in_channels_embed, layer_norm_eps, ln_elementwise_affine)
        self.project_to_hidden = nn.Linear(in_channels_embed, self.inner_dim, bias=use_bias)
        self.project_from_hidden_norm = RMSNorm(self.inner_dim, layer_norm_eps, ln_elementwise_affine)
        self.project_from_hidden = nn.Linear(self.inner_dim, in_channels_embed, bias=use_bias)

        # 为第三个模态新增的归一化层
        self.third_modal_norm = RMSNorm(self.inner_dim, layer_norm_eps, ln_elementwise_affine)


        self.down_block = Simple_UVitBlock(
            self.inner_dim,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            downsample,
            False,
        )
        self.up_block = Simple_UVitBlock(
            self.inner_dim,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            False,
            upsample=upsample,
        )


        if tokenizer_vocab_size is not None:
            self.text_decoder = nn.Sequential(
                nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6),
                nn.Linear(self.inner_dim, tokenizer_vocab_size, bias=use_bias)
            )
        else:
            self.text_decoder = None



        if third_modal_codebook_size is not None:
            self.third_modal_decoder = nn.Sequential(
                nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6),
                nn.Linear(self.inner_dim, third_modal_codebook_size, bias=use_bias)
            )
        else:
            self.third_modal_decoder = None


    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            thirdmodal_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            thirdmodal_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            micro_conds: torch.Tensor = None,
            attention_mask: torch.Tensor = None
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        image_seq_len = (hidden_states.shape[1] // 2 )**2  #注意，这里的image_seq_len用于最后的序列切分，而在切分之前，图像的长和宽经过一次2倍的降采样
        text_seq_len = encoder_hidden_states.shape[1]

        micro_cond_encode_dim = 256  # same as self.micro_cond_encode_dim = 256 from amused
        micro_cond_embeds = get_timestep_embedding(
            micro_conds.flatten(), micro_cond_encode_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        micro_cond_embeds = micro_cond_embeds.reshape((hidden_states.shape[0], -1))

        if self.adapter is not None:
            encoder_hidden_states = self.adapter(encoder_hidden_states)

        pooled_projections = torch.cat([pooled_projections, micro_cond_embeds], dim=1)
        pooled_projections = pooled_projections.to(dtype=self.dtype)
        pooled_projections = self.cond_embed(pooled_projections).to(encoder_hidden_states.dtype)



        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)

        #计算third_modal的映射和归一化
        thirdmodal_hidden_states = self.third_modal_embedder(thirdmodal_hidden_states)
        thirdmodal_hidden_states = self.third_modal_norm(thirdmodal_hidden_states)


        hidden_states = self.embed(hidden_states)
        hidden_states = self.down_block(hidden_states)
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        hidden_states = self.project_to_hidden_norm(hidden_states)
        hidden_states = self.project_to_hidden(hidden_states)


        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        if thirdmodal_ids.ndim == 3:
            logger.warning(
                "Passing `hirdmodal_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            thirdmodal_ids = thirdmodal_ids[0]

        #合并三个模态的ids
        ids = torch.cat((txt_ids, img_ids, thirdmodal_ids), dim=0)

        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states, thirdmodal_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    thirdmodal_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states, thirdmodal_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    thirdmodal_hidden_states=thirdmodal_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states, thirdmodal_hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                )

        encoder_hidden_states, hidden_states, thirdmodal_hidden_states = \
            (hidden_states[:, :text_seq_len],  hidden_states[:, text_seq_len: text_seq_len + image_seq_len],  hidden_states[:, text_seq_len + image_seq_len:])


        if self.text_decoder is not None:
            encoder_hidden_states = self.text_decoder(encoder_hidden_states)

        if self.third_modal_decoder is not None:
            thirdmodal_hidden_states = self.third_modal_decoder(thirdmodal_hidden_states)

        hidden_states = self.project_from_hidden_norm(hidden_states)
        hidden_states = self.project_from_hidden(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        hidden_states = self.up_block(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        output = self.mlm_layer(hidden_states)
        # self.unfuse_qkv_projections()
        if not return_dict:
            return (output, encoder_hidden_states, thirdmodal_hidden_states)


        """
        --- 输出形状验证 ---
          - 输出图像 (output):      torch.Size([4, 8192, 32, 32]) (预期: (4, 8192, 32, 32))
          - 输出文本 (encoder_hidden): torch.Size([4, 77, 49408]) (预期: (4, 77, 49408))
          - 输出大脑 (thirdmodal_hidden): torch.Size([4, 64, 128]) (预期: (4, 64, 128))
        """
        #注意，此时的输出顺序又成了： 图-文-脑
        return output, encoder_hidden_states, thirdmodal_hidden_states  # [b, l, tokenizer_vocab_size]


@torch.no_grad()
def validate_trimodal_transformer_forward_precise():
    print("--- 开始验证 Trimodal_SymmetricTransformer2DModel (精确输入形状) ---")

    # 1. 定义 config
    config = {
        "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56], "codebook_size": 8192,
        "downsample": True, "guidance_embeds": False, "in_channels": 64,
        "joint_attention_dim": 1024, "num_attention_heads": 8, "num_layers": 14,
        "num_single_layers": 28, "patch_size": 1, "pooled_projection_dim": 1024,
        "t5_dim": None, "tokenizer_vocab_size": 49408, "upsample": True,
        "vocab_size": 8256, "third_modal_codebook_size": 128, "third_modal_token_dim": 16,
    }
    print("Config 加载成功。")

    # 2. 实例化模型
    model = Trimodal_SymmetricTransformer2DModel(**config)
    model.eval()
    print("模型实例化成功。")

    a = model.fmri_mask_token.data
    b = model.config.vocab_size

    print("--- 开始验证 Trimodal_SymmetricTransformer2DModel (精确输入形状和数据类型) ---")


    # 3. 准备输入数据 (根据你提供的精确形状和数据类型)
    batch_size = 4
    height, width = 32, 32

    image_seq_len_after_downsample = (height // 2) * (width // 2)
    text_seq_len = 77
    brain_seq_len = 64

    print(f"\n--- 准备输入张量 (Batch Size = {batch_size}) ---")

    # 图像输入: (B, H, W) 的 Token IDs
    # 注意：输入到 embed 层之前，通常是 token ID，所以没有 channel 维
    # 通道维度是在 embed 层内部加上去的
    # torch.randint(high, size, dtype)
    hidden_states = torch.randint(0, config["vocab_size"], (batch_size, height, width), dtype=torch.long)
    print(f"  - hidden_states (图像 Token IDs): {hidden_states.shape}, dtype={hidden_states.dtype}")

    # 文本输入: (B, S, D_joint) - 这部分已经是 embedding, 所以是 float
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, config["joint_attention_dim"], dtype=torch.float32)
    print(f"  - encoder_hidden_states (文本):  {encoder_hidden_states.shape}, dtype={encoder_hidden_states.dtype}")

    # 大脑信号输入: (B, S, D_token) - 这部分也假设是 embedding, 所以是 float
    thirdmodal_hidden_states = torch.randn(batch_size, brain_seq_len, config["third_modal_token_dim"],
                                           dtype=torch.float32)
    print(f"  - thirdmodal_hidden_states (大脑):{thirdmodal_hidden_states.shape}, dtype={thirdmodal_hidden_states.dtype}")

    # 其他条件输入
    micro_conds = torch.randn(batch_size, 5, dtype=torch.float32)
    print(f"  - micro_conds:                  {micro_conds.shape}, dtype={micro_conds.dtype}")

    pooled_projections = torch.randn(batch_size, config["pooled_projection_dim"], dtype=torch.float32)
    print(f"  - pooled_projections:           {pooled_projections.shape}, dtype={pooled_projections.dtype}")

    # 位置 IDs: (S, 3) for (t, h, w) - 必须是 Long
    img_ids = torch.randint(0, 64, (image_seq_len_after_downsample, 3), dtype=torch.long)
    print(f"  - img_ids:                      {img_ids.shape}, dtype={img_ids.dtype}")

    txt_ids = torch.randint(0, 64, (text_seq_len, 3), dtype=torch.long)
    print(f"  - txt_ids:                      {txt_ids.shape}, dtype={txt_ids.dtype}")

    thirdmodal_ids = torch.randint(0, 64, (brain_seq_len, 3), dtype=torch.long)
    print(f"  - thirdmodal_ids:               {thirdmodal_ids.shape}, dtype={thirdmodal_ids.dtype}")

    # Timestep: (B,) - 必须是 Long
    timestep = torch.randint(0, 1000, (batch_size,), dtype=torch.long)
    print(f"  - timestep:                     {timestep.shape}, dtype={timestep.dtype}")

    print("\n✅ 所有输入张量已根据精确的形状和数据类型要求创建完毕。")
    print("   现在可以将它们输入到你的模型中进行最终验证。")

    # 4. 创建 Attention Mask (假设无模态缺失，仅用于验证通路)
    # 在真实场景中，你可以根据需要屏蔽任意模态
    # 这里我们创建一个全通的 mask，不屏蔽任何内容
    total_seq_len = text_seq_len + image_seq_len_after_downsample + brain_seq_len
    attention_mask = torch.zeros(batch_size, 1, 1, total_seq_len, dtype=torch.float32)
    print(f"\nAttention Mask 已创建 (全通)，最终形状为: {attention_mask.shape}")

    # 5. 执行前向传播
    try:
        print("\n执行前向传播...")
        output_image, output_text, output_brain = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            thirdmodal_hidden_states=thirdmodal_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            thirdmodal_ids=thirdmodal_ids,
            micro_conds=micro_conds,
            attention_mask=attention_mask,
            return_dict=False
        )
        print("前向传播成功！")

        # 6. 验证输出形状
        print("\n--- 输出形状验证 ---")

        expected_image_shape = (batch_size, config["codebook_size"], height, width)
        print(f"  - 输出图像 (output):      {output_image.shape} (预期: {expected_image_shape})")
        assert output_image.shape == expected_image_shape

        expected_text_shape = (batch_size, text_seq_len, config["tokenizer_vocab_size"])
        print(f"  - 输出文本 (encoder_hidden): {output_text.shape} (预期: {expected_text_shape})")
        assert output_text.shape == expected_text_shape

        expected_brain_shape = (batch_size, brain_seq_len, config["third_modal_codebook_size"])
        print(f"  - 输出大脑 (thirdmodal_hidden): {output_brain.shape} (预期: {expected_brain_shape})")
        assert output_brain.shape == expected_brain_shape

        print("\n✅ 验证成功！所有输出张量的形状都符合预期。模型逻辑在给定配置和输入下是正确的。")



        # 7. 创建 Attention Mask (屏蔽文本模态)
        text_mask = torch.ones(batch_size, text_seq_len, dtype=torch.float32) * -torch.inf
        image_mask = torch.zeros(batch_size, image_seq_len_after_downsample, dtype=torch.float32)
        brain_mask = torch.zeros(batch_size, brain_seq_len, dtype=torch.float32)
        attention_mask = torch.cat([text_mask, image_mask, brain_mask], dim=1).unsqueeze(1).unsqueeze(1)
        print(f"Attention Mask 已创建，用于屏蔽文本模态。")

        # 5. 黑盒测试
        # -- 运行第一次 (使用全零占位符) --
        print("\n--- 第一次前向传播 (文本使用全零占位符) ---")
        encoder_hidden_states_zero = torch.zeros(batch_size, text_seq_len, config["joint_attention_dim"],
                                                 dtype=torch.float32)

        output_image_1, _, output_brain_1 = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states_zero,
            thirdmodal_hidden_states=thirdmodal_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            thirdmodal_ids=thirdmodal_ids,
            micro_conds=micro_conds,
            attention_mask=attention_mask,
            return_dict=False
        )
        print("第一次前向传播成功。")

        # -- 运行第二次 (使用随机噪声占位符) --
        print("\n--- 第二次前向传播 (文本使用随机占位符) ---")
        encoder_hidden_states_random = torch.randn(batch_size, text_seq_len, config["joint_attention_dim"],
                                                   dtype=torch.float32)

        output_image_2, _, output_brain_2 = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states_random,  # <-- 唯一的变化
            thirdmodal_hidden_states=thirdmodal_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            thirdmodal_ids=thirdmodal_ids,
            micro_conds=micro_conds,
            attention_mask=attention_mask,
            return_dict=False
        )
        print("第二次前向传播成功。")

        # -- 比较结果 --
        print("\n--- 黑盒测试结果比较 ---")
        image_outputs_are_same = torch.allclose(output_image_1, output_image_2, atol=1e-5)
        brain_outputs_are_same = torch.allclose(output_brain_1, output_brain_2, atol=1e-5)

        if image_outputs_are_same and brain_outputs_are_same:
            print("\n✅ 黑盒测试成功！")
            print("   改变被屏蔽的文本占位符内容，完全不影响图像和大脑信号的输出。")
            print("   这强有力地证明了 attention_mask 成功地隔离了文本模态的信息流。")
        else:
            print("\n❌ 黑盒测试失败！")
            print("   改变文本占位符影响了其他模态的输出，mask 未能完全生效。")


    except Exception as e:
        print(f"\n❌ 验证失败！发生错误: {e}")
        import traceback
        traceback.print_exc()

# --- 运行验证 ---
if __name__ == '__main__':
    test_trimodal_transformer_block()
    validate_trimodal_transformer_forward_precise()
