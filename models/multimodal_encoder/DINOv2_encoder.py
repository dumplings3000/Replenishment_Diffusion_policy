import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class VisionEncoder(nn.Module):
    def __init__(self, vision_tower, feature_select='patch'):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_feature = feature_select
     
        self.load_model()
        
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        print('Loading {} model...'.format(self.vision_tower_name))
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        # self.image_processor.size['shortest_edge'] = 518
        # self.image_processor.crop_size['height'] = 518
        # self.image_processor.crop_size['width'] = 518
        # print(f"shortest_edge: {self.image_processor.size['shortest_edge']}, crop_size: {self.image_processor.crop_size['height']}x{self.image_processor.crop_size['width']}")
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.eval()

        self.is_loaded = True

    def feature_select(self, image_forward_outs, select_feature='patch'):
        if select_feature == 'patch':
            image_features = image_forward_outs.last_hidden_state[:, 1:, :]
            self.select_feature = 'patch'
        elif select_feature == 'cls_patch':
            image_features = image_forward_outs.last_hidden_state[:, 0, :] 
            self.select_feature = 'cls'
        else:
            raise ValueError(f'Unexpected select feature: {select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            print(f"images is a list")
            image_features = []
            cls_img_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                cls_img_features = self.feature_select(image_forward_out, select_feature='cls_patch').to(image.dtype)
                image_features.append(image_feature)
                cls_img_features.append(cls_img_features)
        else:  
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            cls_img_features = self.feature_select(image_forward_outs, select_feature='cls_patch').to(images.dtype)

        return image_features, cls_img_features

    @property
    def dummy_feature(self):
        """Returns a dummy feature tensor with the correct shape."""
        if self.select_feature == 'patch':
            return torch.zeros(1, self.num_patches, self.hidden_size, device=self.device, dtype=self.dtype)
        elif self.select_feature == 'cls_patch':
            return torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            # Fallback, though should not be reached
            return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.config_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.image_processor.crop_size['height'] // self.config.patch_size

    @property
    def num_patches(self):
        return (self.image_processor.crop_size['height'] // self.config.patch_size) ** 2

