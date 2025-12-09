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
from collections import OrderedDict

import torch
import torch.nn as nn

from models.rdp.blocks import (FinalLayer, RDTBlock, TimestepEmbedder, TemporalFuser,
                               get_1d_sincos_pos_embed_from_grid,
                               get_multimodal_cond_pos_embed)


class RDP(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """
    def __init__(
        self,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        vision_cond_len=774,
        # history_cond_len=1542, # <<< MODIFIED: 這個參數不再需要
        fuser_queries=774,       # <<< NEW: TemporalFuser 的輸出 token 數
        fuser_depth=4,           # <<< NEW: TemporalFuser 的層數
        # global_pos_embed_config=None, # <<< MODIFIED: 這個也不再需要
        vision_pos_embed_config=None,
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        # self.history_cond = history_cond_len # <<< MODIFIED: 不再使用
        self.vision_cond = vision_cond_len
        self.dtype = dtype
        # self.global_pos_embed_config = global_pos_embed_config
        self.vision_pos_embed_config = vision_pos_embed_config

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        self.fuser_queries = fuser_queries   #20251027
        
        # We will use trainable sin-cos embeddings
        # [timestep; state; action]
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, horizon+3, hidden_size))
        # history conditions
        # self.history_cond_pos_embed = nn.Parameter(
        #     torch.zeros(1, history_cond_len, hidden_size))
        # Image conditions
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, vision_cond_len, hidden_size))
        
        # 20251027
        self.temporal_fuser = TemporalFuser(
            dim=hidden_size,
            n_queries=fuser_queries,
            num_heads=num_heads,
            depth=fuser_depth,
            qkv_bias=True,  # 推薦
            drop=0.0,
            attn_drop=0.0
        )

        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),
                ('ctrl_freq', 1),
                ('state', 1),
                ('action', self.horizon),
            ])
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # if self.global_pos_embed_config is None:
        #     history_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
        #         self.hidden_size, torch.arange(self.history_cond))
        # else:
        #     history_cond_pos_embed = get_multimodal_cond_pos_embed(
        #         embed_dim=self.hidden_size,
        #         mm_cond_lens=OrderedDict(self.global_pos_embed_config),
        #         embed_modality=False
        #     )
        # self.history_cond_pos_embed.data.copy_(
        #     torch.from_numpy(history_cond_pos_embed).float().unsqueeze(0))
        
        if self.vision_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.vision_cond))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.vision_pos_embed_config),
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
            
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(self, x, freq, t, history_c, img_c, history_mask=None, img_mask=None):
        """
        Forward pass of RDT.
        
        x: (B, T, D), state + action token sequence, T = horizon + 1,
            dimension D is assumed to be the same as the hidden size.
        freq: (B,), a scalar indicating control frequency.
        t: (B,) or (1,), diffusion timesteps.
        history_c: (B, L_history, D) or None, history condition tokens (variable length),
            dimension D is assumed to be the same as the hidden size.
        img_c: (B, L_img, D) or None, image condition tokens (fixed length),
            dimension D is assumed to be the same as the hidden size.
        history_mask: (B, L_history) or None, historyuage condition mask (True for valid).
        img_mask: (B, L_img) or None, image condition mask (True for valid).
        """
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)    # (B, 1, D)
        # Append timestep to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, freq, x], dim=1)               # (B, T+1, D)
        
        # Add multimodal position embeddings
        x = x + self.x_pos_embed
        # Note the history is of variable length
        # history_c = history_c + self.history_cond_pos_embed[:, :history_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed
        history_c= self.temporal_fuser(history_c)
    
        # Forward pass
        # conds = [history_c, img_c1, history_c, img_c2]
        conds = [history_c, img_c]

        masks = [history_mask, img_mask]
        for i, block in enumerate(self.blocks):
            # c, mask = conds[i%4], masks[i%2]
            c, mask = conds[i%2], masks[i%2]
            
            x = block(x, c, mask)                       # (B, T+1, D)
        # Inject the historyuage condition at the final layer
        x = self.final_layer(x)                         # (B, T+1, out_channels)

        # Only preserve the action tokens
        x = x[:, -self.horizon:]
        return x
