import torch
import torch.nn as nn
from easydict import EasyDict
import models.ULIP.models.ULIP_models as models
from models.ULIP.utils.utils import get_model

def PonderV2Processor(pcd):
    return torch.from_numpy(pcd).float() if not isinstance(pcd, torch.Tensor) else pcd

class PointTower(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.is_loaded = False
        
        self._model_path = model_path

        self.pcd_processor = PonderV2Processor

        self.load_model()

    def load_model(self):
        if self.is_loaded:
            print('ULIP model is already loaded, `load_model` called again, skipping.')
            return

        print(f"Loading ULIP2 model from checkpoint: {self._model_path}")
        args = EasyDict({
            'evaluate_3d': True,
        })

        # 2. 建立模型
        self.pcd_encoder = getattr(models, "ULIP_PointBERT")(args=args)
        
        # 3. 載入預訓練權重
        ckpt = torch.load(self._model_path, map_location='cpu')
        
        # 從 checkpoint 中提取模型狀態字典 (通常在 'model' 或 'state_dict' key 中)
        state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        
        self.pcd_encoder.load_state_dict(state_dict, strict=True)
        self.pcd_encoder.eval() # 設置為評估模式
        print(f"ULIP2 model loaded successfully.")
        self.is_loaded = True

    def feature_select(self, image_forward_outs, select_feature='patch'):
        if select_feature == 'patch':
            image_features = image_forward_outs[:, 1:, :]
        elif select_feature == 'cls_patch':
            image_features = image_forward_outs[:, 0, :] 
        else:
            raise ValueError(f'Unexpected select feature: {select_feature}')
        return image_features


    @torch.no_grad()
    def forward(self, point_clouds):
        pcd_forward_outs, origin_pcd_features = get_model(self.pcd_encoder).encode_pc(point_clouds.to(self.device, dtype=torch.float32))
        pcd_features = pcd_forward_outs / pcd_forward_outs.norm(dim=-1, keepdim=True)
        cls = self.feature_select(origin_pcd_features, select_feature='cls_patch')
        patch = self.feature_select(origin_pcd_features, select_feature='patch')
        return pcd_features.to(point_clouds.dtype), cls.to(point_clouds.dtype), patch.to(point_clouds.dtype)

    @property
    def dtype(self):
        return next(self.pcd_encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.pcd_encoder.parameters()).device
    
    @property
    def config(self):
        return self.pcd_encoder.config

    @property
    def global_hidden_size(self):
        return self.pcd_encoder.pc_projection.shape[1]
    
    @property
    def hidden_size(self):
        return self.pcd_encoder.point_encoder.trans_dim
    
    @property
    def num_patches(self):
        return 512
        # return self.config.num_patches if hasattr(self.config, 'num_patches') else 512
