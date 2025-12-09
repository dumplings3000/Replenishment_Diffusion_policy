from collections import OrderedDict

import torch
import torch.nn as nn

from timm.models.vision_transformer import Attention, Mlp, RmsNorm
from models.fusion.blocks import TransformerEncoder, CrossAttention

def _module_init(module):
    """標準的 Transformer 權重初始化 (使用 Xavier for Linear, Normal for Embeds)"""
    if isinstance(module, nn.Linear):
        # 這是 Transformer 的標準實踐，對線性層使用 Xavier
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        # Norm 層的權重通常初始化為 1.0，bias 初始化為 0
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Parameter) and module.dim() > 1:
        # 對於可學習的嵌入，使用標準的正態分佈 (std=0.02)
        nn.init.normal_(module, std=.02)

class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim = 4):
        super(FiLM, self).__init__()
        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)

        self.apply(_module_init)
        
    def forward(self, x, condition):
        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)        
        y = gamma * x + beta 
        return y

class ImageFusionModel(nn.Module):
    def __init__(self, 
                 dim=1536,
                 depth=12,
                 drop_path_rate=0.1,
                 num_heads=8,
                 img_token_len = 256,
                 ):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = TransformerEncoder(
            embed_dim=dim,
            depth=depth,
            drop_path_rate=dpr,
            num_heads=num_heads
        )
        self.norm = nn.LayerNorm(dim)
        self.pos_embed = nn.Parameter(torch.randn(1, img_token_len + 1, dim))
        self.projection_head = nn.Linear(dim * 2, dim)

        self.apply(_module_init)
        # print(f"ImageFusionModel: dim={dim}, depth={depth}, drop_path_rate={drop_path_rate}, num_heads={num_heads}, img_token_len={img_token_len}")

    def forward(self, x):
        x += self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        return self.projection_head(x)

class VisionFusionModel(nn.Module):
    def __init__(self,
                 hidden_size=1152,
                 pcd_token_len=514,
                 img_token_len=(258*3),
                 num_heads=16,
                 **block_kwargs
                ):
        super().__init__()

        self.pcd_token_len = pcd_token_len
        self.img_token_len = img_token_len
        # Positional embedding for vision tokens
        self.pcd_cond_pos_embed = nn.Parameter(
            torch.zeros(1, self.pcd_token_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, self.img_token_len, hidden_size))
        self.modality_embed_img = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.modality_embed_pcd = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # Img feature extractor
        self.img_norm1 = RmsNorm(hidden_size, eps=1e-5)
        self.img_attn = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.img_norm2 = RmsNorm(hidden_size, eps=1e-5)
        self.img_cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.img_norm3 = RmsNorm(hidden_size, eps=1e-5)
        self.img_attn2 = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)

        self.img_norm4 = RmsNorm(hidden_size, eps=1e-5)
        self.img_cross_attn2 = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.img_norm5 = RmsNorm(hidden_size, eps=1e-5)
        self.img_ffn = Mlp(in_features=hidden_size, 
            hidden_features=hidden_size, 
            act_layer=approx_gelu, drop=0)
        # Pcd feature extractor
        self.pcd_norm1 = RmsNorm(hidden_size, eps=1e-5)
        self.pcd_attn = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.pcd_norm2 = RmsNorm(hidden_size, eps=1e-5)
        self.pcd_cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.pcd_norm3 = RmsNorm(hidden_size, eps=1e-5)
        self.pcd_attn2 = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.pcd_norm4 = RmsNorm(hidden_size, eps=1e-5)
        self.pcd_cross_attn2 = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.pcd_norm5 = RmsNorm(hidden_size, eps=1e-5)
        self.pcd_ffn = Mlp(in_features=hidden_size, 
            hidden_features=hidden_size, 
            act_layer=approx_gelu, drop=0)
        # Fusion layer
        self.final_norm1 = RmsNorm(hidden_size, eps=1e-5)
        self.final_attn = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.apply(_module_init)

    def forward(self, img, pcd, img_mask = None, pcd_mask = None):
        O_img = img + self.img_cond_pos_embed
        origin_img = O_img
        O_pcd = pcd +self.pcd_cond_pos_embed
        origin_pcd = O_pcd
        """"""
        img = self.img_norm1(O_img)
        img = self.img_attn(img)
        img = img + origin_img
        
        origin_img = img
        img = self.img_norm2(img)
        img = self.img_cross_attn(img, self.pcd_norm1(O_pcd), img_mask)
        img = img + origin_img
                
        origin_img = img
        img = self.img_norm3(img)
        img = self.img_attn2(img)
        img = img + origin_img
        """"""
        pcd = self.pcd_norm1(O_pcd)
        pcd = self.pcd_attn(pcd)
        pcd = pcd + origin_pcd
        
        origin_pcd = pcd
        pcd = self.pcd_norm2(pcd)
        pcd = self.pcd_cross_attn(pcd, self.img_norm1(O_img), pcd_mask)
        pcd = pcd + origin_pcd
                
        origin_pcd = pcd
        pcd = self.pcd_norm3(pcd)
        pcd = self.pcd_attn2(pcd)
        pcd = pcd + origin_pcd
        """"""
        origin_img = img
        origin_pcd = pcd
        img = self.img_norm4(img)
        pcd = self.pcd_norm4(pcd)

        img = self.img_cross_attn2(img, self.pcd_norm4(origin_pcd), img_mask)
        pcd = self.pcd_cross_attn2(pcd, self.img_norm4(origin_img), pcd_mask)
        img = img + origin_img
        pcd = pcd + origin_pcd

        origin_img = img
        img = self.img_norm5(img)
        img = self.img_ffn(img)
        img = img + origin_img

        origin_pcd = pcd
        pcd = self.pcd_norm5(pcd)
        pcd = self.pcd_ffn(pcd)
        pcd = pcd + origin_pcd

        x = torch.cat([pcd, img], dim=1)
        x[:, :self.pcd_token_len, :] = x[:, :self.pcd_token_len, :] + self.modality_embed_pcd
        x[:, self.pcd_token_len:] = x[:, self.pcd_token_len:, :] + self.modality_embed_img

        origin_x = x
        x = self.final_norm1(x)
        x = self.final_attn(x)
        x = x + origin_x
        
        return x