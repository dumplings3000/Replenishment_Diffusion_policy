import torch
import torch.nn as nn
from pointmlp_model import PointMLP

class PointMLPTower(nn.Module):
    def __init__(self, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.select_feature = getattr(args, 'mm_point_select_feature', 'global')  # 可選 'global' 或 'per_point'

        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            print('PointMLP is already loaded, `load_model` called again, skipping.')
            return

        self.point_encoder = PointMLP()  # 載入 PointMLP 模型
        self.point_encoder.eval()

        self.is_loaded = True

    def feature_select(self, point_forward_outs):
        if self.select_feature == 'global':
            # 假設輸出為 (B, 1024)，可作為 global token
            point_features = point_forward_outs
        elif self.select_feature == 'per_point':
            # 假設可取出 per-point 特徵 (B, N, C)
            point_features = point_forward_outs.per_point_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return point_features

    @torch.no_grad()
    def forward(self, point_clouds):
        if type(point_clouds) is list:
            point_features = []
            for pc in point_clouds:
                pc = pc.to(device=self.device, dtype=self.dtype).unsqueeze(0)  # (1, N, 3)
                point_forward_out = self.point_encoder(pc)
                point_feature = self.feature_select(point_forward_out).to(pc.dtype)
                point_features.append(point_feature)
        else:
            point_forward_outs = self.point_encoder(point_clouds.to(device=self.device, dtype=self.dtype))  # (B, N, 3)
            point_features = self.feature_select(point_forward_outs).to(point_clouds.dtype)

        return point_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.point_encoder.dtype if hasattr(self.point_encoder, 'dtype') else torch.float32

    @property
    def device(self):
        return next(self.point_encoder.parameters()).device

    @property
    def hidden_size(self):
        return 1024  # 根據 PointMLP 輸出特徵維度調整
