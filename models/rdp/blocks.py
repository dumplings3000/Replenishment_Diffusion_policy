# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn


#################################################################################
#               Embedding Layers for Timesteps and Condition Inptus             #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                          Cross Attention Layers                               #
#################################################################################
class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim,
            num_heads = 8,
            qkv_bias = False,
            qk_norm = False,
            attn_drop = 0,
            proj_drop = 0,
            norm_layer = nn.LayerNorm,
            eps = 1e-5,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.q_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, c, mask = None):
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))

            attn = torch.clamp(attn, min=-10.0, max=10.0)
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


#################################################################################
#                                 RDT Block                                     #
#################################################################################
class RDTBlock(nn.Module):
    """
    A RDT block with cross-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-5)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.norm2 = RmsNorm(hidden_size, eps=1e-5)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(in_features=hidden_size, 
            hidden_features=hidden_size, 
            act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-5)

    def forward(self, x, c, mask=None):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x
        
        origin_x = x
        x = self.norm2(x)
        x = self.cross_attn(x, c, mask)
        x = x + origin_x
                
        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-5)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim,))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, 
                                  embed_modality=True):
    """
    Generate position embeddings for multimodal conditions. 
    
    mm_cond_lens: an OrderedDict containing 
        (modality name, modality token length) pairs.
        For `"image"` modality, the value can be a multi-dimensional tuple.
        If the length < 0, it means there is no position embedding for the modality or grid.
    embed_modality: whether to embed the modality information. Default is True.
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    if embed_modality:
        # Get embeddings for various modalites
        # We put it in the first half
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # The second half is for position embeddings
        pos_embed_dim = embed_dim // 2
    else:
        # The whole embedding is for position embeddings
        pos_embed_dim = embed_dim
    
    # Get embeddings for positions inside each modality
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "vision" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(
                pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim,))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(
                pos_embed_dim, torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)
    
    return c_pos_emb


#################################################################################
#                               history Functions                               #
#################################################################################
class TemporalFusionBlock(nn.Module):
    """
    一個專門用於融合時序特徵的 Transformer Block。
    它包含:
    1. Cross-Attention (讓 Queries "讀取" 拼接後的歷史 Context)
    2. Self-Attention (讓 Queries 之間 "交流" 讀取到的資訊)
    3. MLP (FFN)
    
    這個架構受到 RDTBlock 的啟發，使用 Pre-Normalization (RmsNorm)。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        **block_kwargs
    ):
        super().__init__()
        
        # 1. Cross-Attention 部分
        # (使用你檔案中定義的 CrossAttention)
        self.norm_q_cross = RmsNorm(dim, eps=1e-5)
        self.norm_kv_cross = RmsNorm(dim, eps=1e-5)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, qk_norm=True, 
            norm_layer=RmsNorm, **block_kwargs
        )
        
        # 2. Self-Attention 部分
        # (使用 timm.models.vision_transformer.Attention)
        self.norm_q_self = RmsNorm(dim, eps=1e-5)
        self.self_attn = Attention(
            dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, qk_norm=True, 
            norm_layer=RmsNorm, **block_kwargs
        )
        
        # 3. MLP 部分
        # (使用 timm.models.vision_transformer.Mlp)
        self.norm_mlp = RmsNorm(dim, eps=1e-5)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu, 
            drop=block_kwargs.get('proj_drop', 0)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q: (b, n_queries, dim) - 我們的可學習 queries
        kv: (b, T * 1288, dim) - 我們的歷史 context (c1 + c2)
        """
        # 1. Cross-Attention (Queries "讀取" Context) + 殘差
        # 你的 CrossAttention 接收 (x, c)
        q = q + self.cross_attn(self.norm_q_cross(q), self.norm_kv_cross(kv))
        
        # 2. Self-Attention (Queries 之間 "交流") + 殘差
        # timm 的 Attention 接收 (x)
        q = q + self.self_attn(self.norm_q_self(q))
        
        # 3. MLP + 殘差
        q = q + self.mlp(self.norm_mlp(q))
        
        return q


class TemporalFuser(nn.Module):
    """
    時序融合器 (Temporal Fuser)
    
    接收 (b*2, 1288, dim) 的 history_c，
    並輸出 (b, n_queries, dim) 的融合後條件。
    這個輸出可以直接作為 RDTBlock 的輸入 `c`。
    """
    def __init__(
        self,
        dim: int,
        n_queries: int = 128,    # 最終輸出的 token 長度 (例如 128 或 256)
        num_heads: int = 8,
        depth: int = 2,          # 融合 block 的層數 (建議 2-4)
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.
    ):
        super().__init__()
        
        # 1. 可學習的時序編碼 (區分 t-1 和 t)
        # 索引 0: t-1 (history_c1)
        # 索引 1: t   (history_c2)
        self.temporal_embed = nn.Embedding(2, dim)

        # 2. 可學習的 Queries (Q)
        # 這些 queries 的任務是去 "閱讀" 和 "總結" 所有的歷史特徵
        self.learnable_queries = nn.Parameter(torch.randn(1, n_queries, dim))
        nn.init.normal_(self.learnable_queries, std=.02) # 推薦使用標準初始化

        # 準備傳遞給 Block 的參數
        block_kwargs = {
            'proj_drop': drop, 
            'attn_drop': attn_drop
        }
        
        # 3. 堆疊多層 TemporalFusionBlock
        self.blocks = nn.ModuleList([
            TemporalFusionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                **block_kwargs
            )
            for _ in range(depth)
        ])
        
        # 4. 最終的輸出 Norm (風格同 FinalLayer)
        self.norm_out = RmsNorm(dim, eps=1e-5)

    def forward(self, history_c: torch.Tensor) -> torch.Tensor:
        """
        輸入 history_c: (b*2, 1288, dim)
        輸出 fused_condition: (b, n_queries, dim)
        """
        
        # -----------------------------------------------------------
        # 這裡是你提供的起始程式碼
        # history_c shape: (b*2, 1288, dim)
        history_c1, history_c2 = torch.chunk(history_c, 2, dim=0)
        # history_c1 shape: (b, 1288, dim)  (<- 這是 t-1)
        # history_c2 shape: (b, 1288, dim)  (<- 這是 t)
        # -----------------------------------------------------------

        b, n, c = history_c1.shape
        device = history_c1.device

        # --- 步驟 1: 添加時序編碼 ---
        # 獲取 t-1 (index 0) 的編碼, shape: (1, 1, dim)
        t_emb1 = self.temporal_embed(torch.tensor(0, device=device)).view(1, 1, c)
        # 獲取 t (index 1) 的編碼, shape: (1, 1, dim)
        t_emb2 = self.temporal_embed(torch.tensor(1, device=device)).view(1, 1, c)

        # 廣播並添加到 1288 個 token 上
        history_c1_encoded = history_c1 + t_emb1
        history_c2_encoded = history_c2 + t_emb2
        
        # --- 步驟 2: 拼接 Context (K, V) ---
        # 沿著 "token" 維度拼接
        # context shape: (b, 1288 + 1288, dim) -> (b, 2576, dim)
        context = torch.cat([history_c1_encoded, history_c2_encoded], dim=1)
        
        # --- 步驟 3: 準備 Queries (Q) ---
        # 擴展 learnable_queries 以匹配 batch size
        # queries shape: (b, n_queries, dim)
        queries = self.learnable_queries.expand(b, -1, -1)
        
        # --- 步驟 4: 執行融合 (依序通過所有 blocks) ---
        for block in self.blocks:
            queries = block(queries, context)
            
        # --- 步驟 5: 最終 Norm ---
        fused_condition = self.norm_out(queries)
        
        # 最終輸出的 shape: (b, n_queries, dim), 例如 (b, 128, dim)
        return fused_condition