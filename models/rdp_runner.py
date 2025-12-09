import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdp.model import RDP
from models.fusion.model import VisionFusionModel, ImageFusionModel, FiLM
from timm.models.vision_transformer import RmsNorm


class model_Runner(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(self, *, action_dim, pred_horizon, config, 
                 img_token_dim, pcd_token_dim, global_pcd_dim, 
                 state_token_dim, vision_token_dim,
                 img_token_len, pcd_token_len,
                 vision_cond_len, 
                #  history_cond_len, global_pos_embed_config=None, 
                 vision_pos_embed_config=None, 
                 num_cam=3,
                 dtype=torch.bfloat16):
        super(model_Runner, self).__init__()

        self.group1_loaded_param_ids = set() # (G1, Low LR) "原本凍結的"
        self.all_loaded_param_ids = set()    # (G1 + G2) "所有在權重檔的"

        self.hidden_size = config['rdp']['hidden_size']
        # self.history_cond_len = history_cond_len
        self.num_cam = num_cam

        self.imgfusionmodel = ImageFusionModel(
            dim=img_token_dim,
            img_token_len = img_token_len,
        )
        self.film = FiLM(
            vision_token_dim
        )

        self.Vision_model = VisionFusionModel(
            hidden_size = vision_token_dim,
            pcd_token_len=pcd_token_len+2,
            img_token_len=(img_token_len+2)*self.num_cam,
        )

        self.backbone_model = RDP(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=self.hidden_size,
            depth=config['rdp']['depth'],
            num_heads=config['rdp']['num_heads'],
            vision_cond_len=vision_cond_len,
            # history_cond_len=history_cond_len,
            vision_pos_embed_config=vision_pos_embed_config,
            # global_pos_embed_config=global_pos_embed_config,
            dtype=dtype,
        )
        """
        ================================================Create adpators for inputs================================================
        """
        # Create adpators for various conditional inputs
        self.pcd_adaptor = self.build_condition_adapter(
            config['pcd_adaptor'], 
            in_features=pcd_token_dim, 
            out_features=global_pcd_dim
        )
        self.pcd_embed_adaptor = self.build_condition_adapter(
            config['pcd_embed_adaptor'], 
            in_features=global_pcd_dim, 
            out_features=vision_token_dim
        )
        self.img_embed_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=vision_token_dim
        )
        self.vision_adaptor = self.build_condition_adapter(
            config['vision_adaptor'], 
            in_features=vision_token_dim, 
            out_features=self.hidden_size
        )
        self.vision_adaptor_norm = RmsNorm(self.hidden_size, eps=1e-5)
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=state_token_dim * 2,    # state + state mask (indicator)
            out_features=self.hidden_size
        )
        
        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        ) 
        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # print("Diffusion params: %e" % sum(
        #     [p.numel() for p in self.model.parameters()] + 
        #     [p.numel() for p in self.lang_adaptor.parameters()] + 
        #     [p.numel() for p in self.img_adaptor.parameters()] + 
        #     [p.numel() for p in self.state_adaptor.parameters()]))
        
    def load_model(self, model_path):
        """
        Load the model from the given path.
        """
        print(f"Loading model from {model_path}")
        ckpt = torch.load(model_path, map_location='cpu')
        # self.full_state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
        # if "module" in ckpt:
        #     self.full_state_dict = ckpt["module"]
        # else:
        #     self.full_state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
        if "module" in ckpt:
            full_state_dict = ckpt["module"]
        else:
            full_state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

        # ==========================================================
        # !! 核心修改部分：僅針對您指定的五個前綴進行鍵名替換 !!
        # ==========================================================
        prefixes_to_convert = (
            'model.blocks',
            'model.final_layer',
            'model.freq_embedder',
            'model.t_embedder',
            'model.x_pos_embed' 
        )
        
        new_state_dict = OrderedDict()
        
        # 遍歷原 state_dict 中的所有鍵
        for key, value in full_state_dict.items():
            
            # 檢查鍵名是否以任一個目標前綴開頭
            if any(key.startswith(p) for p in prefixes_to_convert):
                # 執行替換：'model.' -> 'backbone_model.'
                new_key = key.replace('model.', 'backbone_model.', 1)
                new_state_dict[new_key] = value
            else:
                # 其他鍵（包括其他 'model.' 開頭的鍵，若有）保持不變
                new_state_dict[key] = value

        # 用轉換後的字典替換原字典
        full_state_dict = new_state_dict
        print("Model key names updated: 'model.' -> 'backbone_model.' for specified prefixes.")
        # ==========================================================

        # for key in list(self.full_state_dict.keys()):
        #     print(key)
        prefixes_to_load = (
            'backbone_model.blocks',
            'backbone_model.final_layer',
            'backbone_model.freq_embedder',
            'backbone_model.t_embedder',
            'backbone_model.x_pos_embed' 
        )
        # prefixes_to_load = (
        #     'model.blocks',
        #     'model.final_layer',
        #     'model.freq_embedder',
        #     'model.t_embedder',
        #     'model.x_pos_embed' 
        # )

        state_prefix = (
            'state_adaptor'
        )
        # g1_p1_ids = self.parse_weight(self.backbone_model, prefixes_to_load, remove_prefix="backbone_model.")
        # g1_p2_ids = self.parse_weight(self.state_adaptor, state_prefix, remove_prefix="state_adaptor.")
        # self.group1_loaded_param_ids = g1_p1_ids.union(g1_p2_ids)
        # all_ids = set()
        # all_ids.update(self.parse_weight(self.backbone_model, ('backbone_model.',), 'backbone_model.'))
        # all_ids.update(self.parse_weight(self.state_adaptor, ('state_adaptor.',), 'state_adaptor.')) # G1 state_adaptor
        # all_ids.update(self.parse_weight(self.imgfusionmodel, ('imgfusionmodel.',), 'imgfusionmodel.'))
        # all_ids.update(self.parse_weight(self.film, ('film.',), 'film.'))
        # all_ids.update(self.parse_weight(self.Vision_model, ('Vision_model.',), 'Vision_model.'))
        # all_ids.update(self.parse_weight(self.pcd_adaptor, ('pcd_adaptor.',), 'pcd_adaptor.'))
        # all_ids.update(self.parse_weight(self.pcd_embed_adaptor, ('pcd_embed_adaptor.',), 'pcd_embed_adaptor.'))
        # all_ids.update(self.parse_weight(self.img_embed_adaptor, ('img_embed_adaptor.',), 'img_embed_adaptor.'))
        # all_ids.update(self.parse_weight(self.vision_adaptor, ('vision_adaptor.',), 'vision_adaptor.'))

        g1_p1_ids = self.parse_weight(self.backbone_model, prefixes_to_load, full_state_dict, remove_prefix="backbone_model.")
        g1_p2_ids = self.parse_weight(self.state_adaptor, state_prefix, full_state_dict, remove_prefix="state_adaptor.")
        self.group1_loaded_param_ids = g1_p1_ids.union(g1_p2_ids)
        
        all_ids = set()
        all_ids.update(self.parse_weight(self.backbone_model, ('backbone_model.',), full_state_dict, 'backbone_model.'))
        all_ids.update(self.parse_weight(self.state_adaptor, ('state_adaptor.',), full_state_dict, 'state_adaptor.'))
        all_ids.update(self.parse_weight(self.imgfusionmodel, ('imgfusionmodel.',), full_state_dict, 'imgfusionmodel.'))
        all_ids.update(self.parse_weight(self.film, ('film.',), full_state_dict, 'film.'))
        all_ids.update(self.parse_weight(self.Vision_model, ('Vision_model.',), full_state_dict, 'Vision_model.'))
        all_ids.update(self.parse_weight(self.pcd_adaptor, ('pcd_adaptor.',), full_state_dict, 'pcd_adaptor.'))
        all_ids.update(self.parse_weight(self.pcd_embed_adaptor, ('pcd_embed_adaptor.',), full_state_dict, 'pcd_embed_adaptor.'))
        all_ids.update(self.parse_weight(self.img_embed_adaptor, ('img_embed_adaptor.',), full_state_dict, 'img_embed_adaptor.'))
        all_ids.update(self.parse_weight(self.vision_adaptor, ('vision_adaptor.',), full_state_dict, 'vision_adaptor.'))

        all_ids.update(self.group1_loaded_param_ids)
        self.all_loaded_param_ids = all_ids
        # g2_count = len(self.all_loaded_param_ids) - len(self.group1_loaded_param_ids)
        # self.freeze_model()

    def parse_weight(self, model, prefix, full_state_dict, remove_prefix="model."): # <-- 接收 full_state_dict
        partial_state_dict = OrderedDict()
        for key, value in full_state_dict.items(): # <-- 使用傳入的 full_state_dict
            if key.startswith(prefix):
                partial_state_dict[key] = value

        corrected_state_dict = OrderedDict()
        for key, value in partial_state_dict.items():
            new_key = key
            if key.startswith(remove_prefix):
                new_key = key.replace(remove_prefix, '', 1)
            corrected_state_dict[new_key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(corrected_state_dict, strict=False)
        
        if unexpected_keys:
            print(f"[Warning] Unexpected keys found in checkpoint: {unexpected_keys}")

        if missing_keys:
            print(f"Missing keys (weights not loaded into the model): {len(missing_keys)}")
            print(f"missing keys: {missing_keys[:5]}") 

        loaded_param_ids = set()
        
        # 獲取这个子模組中 {參數名稱: 參數物件} 的對應字典
        model_params_by_name = dict(model.named_parameters())
        
        # 成功載入的 key = 權重檔有的 key - 權重檔有但模型沒有的 key
        loaded_keys = corrected_state_dict.keys() - set(unexpected_keys)
        
        for key in loaded_keys:
            if key in model_params_by_name:
                # 這個 key 對應到一個 nn.Parameter
                loaded_param_ids.add(id(model_params_by_name[key]))
            
        return loaded_param_ids
    
    def get_parameter_groups(self, lr_low, lr_mid, lr_high):
        """
        將參數分為三組 (根據你的精確邏輯):
        Group 1 (low):   "原本被凍結的" (G1 ID 集合)
        Group 2 (mid):   "權重檔裡的沒被凍的" (G1 之外，但在 'all_loaded' 集合中)
        Group 3 (high):  "沒在權重檔裡的" (所有不在 'all_loaded' 集合中的)
        """
        print(f"Setting up 3 parameter groups (Precise Logic):")
        print(f"  - Group 1 (Low LR):   {lr_low} (Loaded Backbone/State via G1 prefixes)")
        print(f"  - Group 2 (Mid LR):   {lr_mid} (All Other Loaded Params)")
        print(f"  - Group 3 (High LR):  {lr_high} (New/Not-Loaded Params)")
        
        # 確保 ID 集合存在
        if not hasattr(self, 'group1_loaded_param_ids'):
            self.group1_loaded_param_ids = set()
            print("[Warning] Group 1 ID set not found. Was load_model() called?")
        if not hasattr(self, 'all_loaded_param_ids'):
            self.all_loaded_param_ids = set()
            print("[Warning] 'all_loaded' ID set not found. Was load_model() called?")

        group1_params = []
        group2_params = []
        group3_params = []
        
        # !! 新增：用於日誌的名稱列表 !!
        g1_names = []
        g2_names = []
        g3_names = []

        g1_ids = self.group1_loaded_param_ids
        all_loaded_ids = self.all_loaded_param_ids

        # 遍歷模型中的 *所有* 參數
        for name, param in self.named_parameters():
            param.requires_grad = True # 解凍所有參數
            
            param_id = id(param)
            
            if param_id in g1_ids:
                # 優先歸入 G1
                group1_params.append(param)
                g1_names.append(name) # <-- 新增
            elif param_id in all_loaded_ids:
                # 其次，如果在 "all_loaded" 中 (但不在 G1 中)，歸入 G2
                group2_params.append(param)
                g2_names.append(name) # <-- 新增
            else:
                # 剩下的就是 G3 (新增的網路)
                group3_params.append(param)
                g3_names.append(name) # <-- 新增
        
        # 建立優化器所需的字典列表
        parameter_groups = [
            {'params': group1_params, 'lr': lr_low},
            {'params': group2_params, 'lr': lr_mid},
            {'params': group3_params, 'lr': lr_high}
        ]
        
        # # ==============================================================
        # #            !! 新增：詳細日誌輸出 !!
        # # ==============================================================
        # print("\n" + "=" * 70)
        # print("           PARAMETER GROUPING VERIFICATION")
        # print("=" * 70)

        # # 內部輔助函數，用於印出範例
        # def print_group_samples(group_name, names_list, lr):
        #     print(f"\nDetails for {group_name} (LR: {lr}, Total: {len(names_list)})")
        #     if not names_list:
        #         print("  -> (No parameters in this group)")
        #         return

        #     # 印出前 5 個
        #     print("  -> First 5 params:")
        #     for name in names_list[:]:
        #         print(f"     - {name}")
            
        #     # # 印出後 5 個
        #     # if len(names_list) > 5:
        #     #     print("  -> Last 5 params:")
        #     #     for name in names_list[-5:]:
        #     #         print(f"     - {name}")
        
        # print_group_samples("Group 1 (Low LR)", g1_names, lr_low)
        # print("-" * 70)
        # print_group_samples("Group 2 (Mid LR)", g2_names, lr_mid)
        # print("-" * 70)
        # print_group_samples("Group 3 (High LR)", g3_names, lr_high)
        # print("=" * 70 + "\n")
        
        # # 檢查 (你原本的檢查也保留)
        # all_params_count = len(list(self.parameters()))
        # grouped_params_count = len(group1_params) + len(group2_params) + len(group3_params)
        
        # if all_params_count != grouped_params_count:
        #     print(f"[Warning] Parameter count mismatch! All: {all_params_count}, Grouped: {grouped_params_count}")
        # else:
        #     print(f"Successfully grouped {grouped_params_count} parameters into 3 groups.")
            
        # print(f"  - Group 1 (Low LR) param count: {len(group1_params)}")
        # print(f"  - Group 2 (Mid LR) param count: {len(group2_params)}")
        # print(f"  - Group 3 (High LR) param count: {len(group3_params)}")

        # if len(group1_params) == 0:
        #     print("[ERROR] Group 1 has 0 parameters. Check 'load_model' G1 prefixes ('backbone_model.blocks', etc)!")
        # if len(group2_params) == 0:
        #      print("[Warning] Group 2 (Mid LR) has 0 parameters. Check 'load_model' G2 prefixes ('imgfusionmodel.', etc)!")
        # if len(group3_params) == 0:
        #     print("[Warning] Group 3 (High LR) has 0 parameters. This means all model parameters were found in the checkpoint. Is this expected?")

        return parameter_groups
        
    def freeze_model(self):
        """
        Freeze the model weights.
        """
        print("Freezing model weights.")
        for param in self.backbone_model.parameters():
            param.requires_grad = False
        for param in self.state_adaptor.parameters():
            param.requires_grad = False
        print("Model weights frozen.")
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    # ========= Train  ============
    def compute_loss(self, cls_img_embeds, patch_img_embeds, 
                     global_pcd_embeds, cls_pcd_embeds, patch_pcd_embeds,
                     pcd_mean, pcd_scale,
                     state_tokens, action_gt, action_mask, ctrl_freqs,
                     keyframe_cls_img_embeds, keyframe_patch_img_embeds,
                     keyframe_global_pcd_embeds, keyframe_cls_pcd_embeds, keyframe_patch_pcd_embeds,
                     keyframe_pcd_mean, keyframe_pcd_scale,
                    ):
        '''
        cls_img_embeds: (batch_size * img_num, token_len, img_token_dim), (B * N * L, 1, 1536)
        patch_img_embeds: (batch_size * img_num, token_len, img_token_dim), (B * N  *L, 1369, 1536)

        global_pcd_embeds: (batch_size * pcd_num, ,token_len, pcd_embed_dim), (B * L, 1, 512)
        cls_pcd_embeds: (batch_size * pcd_num, token_len, vision_token_dim), (B * L, 1,384)
        patch_pcd_embeds: (batch_size * pcd_num, token_len, vision_token_dim), (B * L, 512, 384)
        pcd_mean: (batch_size * pcd_num, xyz), (B * L, 3)
        pcd_scale: (batch_size * pcd_num), (B * L)

        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = state_tokens.shape[0]
        H = cls_pcd_embeds.shape[0]//batch_size
        device = state_tokens.device  

        # Sample noise that we'll add to the actions
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )

        original_action_mask = action_mask

        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        
        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        
        # Align the dimension with the hidden size
        state_action_traj = self.state_adaptor(state_action_traj)

        # IMAGES Feature concat and align dimension
        global_img_embeds = torch.cat([cls_img_embeds, patch_img_embeds], dim=1)
        global_img_embeds = self.imgfusionmodel(global_img_embeds)
        global_img_embeds = global_img_embeds.unsqueeze(1)
        img_cond = torch.cat([global_img_embeds, cls_img_embeds, patch_img_embeds], dim=1)
        img_cond = self.img_embed_adaptor(img_cond)

        # PCD Feature concat and align dimension 
        raw_pcd_embeds = torch.cat([cls_pcd_embeds, patch_pcd_embeds], dim=1)
        raw_pcd_embeds = self.pcd_adaptor(raw_pcd_embeds)
        pcd_embeds = torch.cat([global_pcd_embeds, raw_pcd_embeds], dim=1)
        pcd_embeds = self.pcd_embed_adaptor(pcd_embeds)

        pcd_film_cond = torch.cat([pcd_mean, pcd_scale], dim=-1)
        pcd_cond = self.film(pcd_embeds, pcd_film_cond) + pcd_embeds

        # Visual cond Feature Fusion
        img_cond = img_cond.reshape(batch_size * H, -1, img_cond.shape[-1])
        vision_cond = self.Vision_model(img_cond, pcd_cond)
        vision_cond = self.vision_adaptor(vision_cond)
        vision_cond = self.vision_adaptor_norm(vision_cond)
        # print(f"vision_cond shape: {vision_cond.shape}, dtype: {vision_cond.dtype}")

        # # global cond Feature Fusion
        global_img_embeds = torch.cat([keyframe_cls_img_embeds, keyframe_patch_img_embeds], dim=1)
        global_img_embeds = self.imgfusionmodel(global_img_embeds)
        global_img_embeds = global_img_embeds.unsqueeze(1)
        keyframe_img_cond = torch.cat([global_img_embeds, keyframe_cls_img_embeds, keyframe_patch_img_embeds], dim=1)
        keyframe_img_cond = self.img_embed_adaptor(keyframe_img_cond)

        raw_pcd_embeds = torch.cat([keyframe_cls_pcd_embeds, keyframe_patch_pcd_embeds], dim=1)
        raw_pcd_embeds = self.pcd_adaptor(raw_pcd_embeds)
        keyframe_pcd_embeds = torch.cat([keyframe_global_pcd_embeds, raw_pcd_embeds], dim=1)
        keyframe_pcd_embeds = self.pcd_embed_adaptor(keyframe_pcd_embeds)

        keyframe_pcd_film_cond = torch.cat([keyframe_pcd_mean, keyframe_pcd_scale], dim=-1)
        keyframe_pcd_cond = self.film(keyframe_pcd_embeds, keyframe_pcd_film_cond) + keyframe_pcd_embeds
        keyframe_img_cond = keyframe_img_cond.reshape(batch_size * H *2, -1, keyframe_img_cond.shape[-1])
        keyframe_vision_cond = self.Vision_model(keyframe_img_cond, keyframe_pcd_cond)
        keyframe_vision_cond = self.vision_adaptor(keyframe_vision_cond)
        keyframe_vision_cond = self.vision_adaptor_norm(keyframe_vision_cond)
        # print(f"keyframe_cond shape: {keyframe_vision_cond.shape}, dtype: {keyframe_vision_cond.dtype}")


        # global_cond = torch.zeros(batch_size, self.history_cond_len, self.hidden_size, device=device).to(dtype=torch.bfloat16)
        # print(f"global_cond shape: {global_cond.shape}, dtype: {global_cond.dtype}")

        # ===================================================================
        # !! 在這裡加入 NaN 檢查哨兵 !!
        # ===================================================================
        if torch.isnan(state_action_traj).any():
            print("[NaN DEBUG] 'state_action_traj' is NaN before backbone_model!")
            raise RuntimeError("NaN DETECTED IN state_action_traj")

        if torch.isnan(keyframe_vision_cond).any():
            print("[NaN DEBUG] 'keyframe_vision_cond' is NaN before backbone_model!")
            raise RuntimeError("NaN DETECTED IN keyframe_vision_cond")

        if torch.isnan(vision_cond).any():
            print("[NaN DEBUG] 'vision_cond' is NaN before backbone_model!")
            raise RuntimeError("NaN DETECTED IN vision_cond")
        
        # ===================================================================
        
        # Predict the denoised result
        pred = self.backbone_model(state_action_traj, ctrl_freqs, 
                          timesteps, keyframe_vision_cond, vision_cond)
        # print(f"pred shape: {pred.shape}")

        pred_type = self.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        loss_unmasked = F.mse_loss(pred, target, reduction='none') 

        # 2. 用你的 original_action_mask [B, 1, 128] 去「過濾」loss
        #    PyTorch 的廣播 (broadcasting) 機制會自動處理 [B, H, 128] * [B, 1, 128]
        loss_masked = loss_unmasked * original_action_mask

        # 3. 只對「7 個真正」的維度取平均 (sum / sum)
        loss = loss_masked.sum() / (original_action_mask.sum() + 1e-5)

        # loss = F.mse_loss(pred, target)
        return loss
    
    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)

    # # ========= Inference  ============
    def conditional_sample(self, keyframe_cond, vision_cond, 
                           state_traj, action_mask, ctrl_freqs):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            model_output = self.backbone_model(state_action_traj, ctrl_freqs,
                                    t.unsqueeze(-1).to(device),
                                    keyframe_cond, # <-- 使用 keyframe_cond
                                    vision_cond)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action
    def predict_action(self, cls_img_embeds, patch_img_embeds, 
                     global_pcd_embeds, cls_pcd_embeds, patch_pcd_embeds,
                     pcd_mean, pcd_scale,
                     keyframe_cls_img_embeds, keyframe_patch_img_embeds,
                     keyframe_global_pcd_embeds, keyframe_cls_pcd_embeds, keyframe_patch_pcd_embeds,
                     keyframe_pcd_mean, keyframe_pcd_scale,
                     state_tokens, action_mask, ctrl_freqs
                    ):
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        
        batch_size = state_tokens.shape[0]
        H = cls_pcd_embeds.shape[0]//batch_size
        device = state_tokens.device 
        
        # Align the dimension with the hidden size
        state_traj = self.state_adaptor(state_tokens)

        # IMAGES Feature concat and align dimension
        global_img_embeds = torch.cat([cls_img_embeds, patch_img_embeds], dim=1)
        global_img_embeds = self.imgfusionmodel(global_img_embeds)
        global_img_embeds = global_img_embeds.unsqueeze(1)
        img_cond = torch.cat([global_img_embeds, cls_img_embeds, patch_img_embeds], dim=1)
        img_cond = self.img_embed_adaptor(img_cond)

        # PCD Feature concat and align dimension 
        raw_pcd_embeds = torch.cat([cls_pcd_embeds, patch_pcd_embeds], dim=1)
        raw_pcd_embeds = self.pcd_adaptor(raw_pcd_embeds)
        pcd_embeds = torch.cat([global_pcd_embeds, raw_pcd_embeds], dim=1)
        pcd_embeds = self.pcd_embed_adaptor(pcd_embeds)

        pcd_film_cond = torch.cat([pcd_mean, pcd_scale], dim=-1)
        pcd_cond = self.film(pcd_embeds, pcd_film_cond) + pcd_embeds

        # Visual cond Feature Fusion
        img_cond = img_cond.reshape(batch_size * H, -1, img_cond.shape[-1])
        vision_cond = self.Vision_model(img_cond, pcd_cond)
        vision_cond = self.vision_adaptor(vision_cond)
        vision_cond = self.vision_adaptor_norm(vision_cond)

        # global cond Feature Fusion
        # ==============================================================
        global_img_embeds = torch.cat([keyframe_cls_img_embeds, keyframe_patch_img_embeds], dim=1)
        global_img_embeds = self.imgfusionmodel(global_img_embeds)
        global_img_embeds = global_img_embeds.unsqueeze(1)
        keyframe_img_cond = torch.cat([global_img_embeds, keyframe_cls_img_embeds, keyframe_patch_img_embeds], dim=1)
        keyframe_img_cond = self.img_embed_adaptor(keyframe_img_cond)

        raw_pcd_embeds = torch.cat([keyframe_cls_pcd_embeds, keyframe_patch_pcd_embeds], dim=1)
        raw_pcd_embeds = self.pcd_adaptor(raw_pcd_embeds)
        keyframe_pcd_embeds = torch.cat([keyframe_global_pcd_embeds, raw_pcd_embeds], dim=1)
        keyframe_pcd_embeds = self.pcd_embed_adaptor(keyframe_pcd_embeds)

        keyframe_pcd_film_cond = torch.cat([keyframe_pcd_mean, keyframe_pcd_scale], dim=-1)
        keyframe_pcd_cond = self.film(keyframe_pcd_embeds, keyframe_pcd_film_cond) + keyframe_pcd_embeds
        
        # 這裡的 H 來自上面的 H (H = cls_pcd_embeds.shape[0]//batch_size)
        # 假設 keyframe 的數量是固定的 (例如 train.py 中的 2)
        keyframe_img_cond = keyframe_img_cond.reshape(batch_size * H *2, -1, keyframe_img_cond.shape[-1])
        keyframe_vision_cond = self.Vision_model(keyframe_img_cond, keyframe_pcd_cond)
        keyframe_vision_cond = self.vision_adaptor(keyframe_vision_cond)
        keyframe_vision_cond = self.vision_adaptor_norm(keyframe_vision_cond)
        # ==============================================================

        # Run sampling
        action_pred = self.conditional_sample(
            keyframe_vision_cond, # <-- 傳入 keyframe_vision_cond
            vision_cond, 
            state_traj, action_mask, ctrl_freqs,
        )
        # # Run sampling
        # action_pred = self.conditional_sample(
        #     global_cond, vision_cond, 
        #     state_traj, action_mask, ctrl_freqs,
        # )
        return action_pred

