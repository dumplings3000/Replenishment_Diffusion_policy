import traceback
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import open3d as o3d
from data.hdf5_final_dataset import HDF5VADataset
from train.image_corrupt import image_corrupt

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class VAConsumerDataset(Dataset):
    """A vision-languange-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """
    
    def __init__(
        self, 
        config,
        image_processor,
        pcd_processor,
        num_cameras,
        img_history_size,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
    ):
        super(VAConsumerDataset, self).__init__()

        # read config
        self.data_pth = config["data_pth"]
        self.image_aspect_ratio = config["image_aspect_ratio"]
        self.pcd_max_num = config["pcd_max_num"]
        self.pcd_min_num = config["pcd_min_num"]
        self.pcd_noise_std = config["pcd_noise_std"]

        self.image_processor = image_processor
        self.pcd_processor = pcd_processor
        
        self.num_cameras = num_cameras
        self.img_history_size = img_history_size

        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug

        self.cond_mask_prob = cond_mask_prob
        self.state_noise_snr = state_noise_snr
        self.cam_ext_mask_prob = cam_ext_mask_prob

        self.hdf5_dataset = None
        self.control_freq = 20
        self.hdf5_dataset = HDF5VADataset(self.data_pth)
        
    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def __len__(self):
        return len(self.hdf5_dataset)
    
    def __getitem__(self, index):
        while True:
            data_dict = None
            try:
                res = self.hdf5_dataset.get_item()
                content = res['meta']
                states = res['state']
                state_std = res['state_std']
                state_mean = res['state_mean']
                state_norm = res['state_norm']
                actions = res['actions']
                state_elem_mask = res['state_indicator']
                img_meta = [
                    res['base_cam'], res['base_cam_mask'],
                    res['hand_cam'], res['hand_cam_mask'],
                    res['side_cam'], res['side_cam_mask'],
                ]
                pcd_meta = [
                    res['pointcloud'], res['pointcloud_mask'],
                ]
                key_frame_meta = res['key_frames_data']
                
                data_dict = {}
                # data_dict['ctrl_freq'] = self.control_freq if random.random() > self.cond_mask_prob else 0
                data_dict['ctrl_freq'] = 15

                if self.state_noise_snr is not None:
                    states += np.random.normal(
                        0.0, state_std / np.sqrt(10 ** (self.state_noise_snr / 10)), 
                        states.shape)
                ds_state_mean = np.tile(state_mean[None], (states.shape[0], 1))
                # Randomly mask the states by the mean state
                data_dict["states"] = states if random.random() > self.cond_mask_prob else ds_state_mean
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = state_elem_mask \
                    if random.random() > self.cond_mask_prob else np.zeros_like(state_elem_mask)
                
                # Stat for the episode that the step belongs to 
                data_dict["state_norm"] = state_norm
                
                # We replace the invalid images with the background image
                # and also randomly mask images by the background image
                background_color = np.array([
                    int(x*255) for x in self.image_processor.image_mean
                ], dtype=np.uint8).reshape(1, 1, 3)
                background_image = np.ones((
                    self.image_processor.crop_size["height"], 
                    self.image_processor.crop_size["width"], 3), dtype=np.uint8
                ) * background_color
                
                img_meta = list(self.pairwise(img_meta))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob
                rearranged_images = []
                for i in range(self.img_history_size):
                    for j in range(self.num_cameras):
                        images, image_mask = img_meta[j]
                        image, valid = images[i], image_mask[i]
                        if valid and (math.prod(image.shape) > 0) and \
                            (random.random() > mask_probs[j]):
                            rearranged_images.append((image, True))
                        else:
                            rearranged_images.append((background_image.copy(), False))
                
                preprocessed_images = []
                for image, valid in rearranged_images:
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image)
                    
                    if valid and self.auto_adjust_image_brightness:
                        img_np = np.array(image)
                        # 正規化到 [0, 1] 並計算平均值
                        average_brightness = img_np.mean() / 255.0
                        if average_brightness <= 0.15:
                        # pixel_values = list(image.getdata())
                        # average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        # if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
                    # Only apply image augmentation to 50% of the images
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice([
                            "corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)
                    
                    if self.image_aspect_ratio == 'pad':
                        image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    preprocessed_images.append(image)
                data_dict["images"] = preprocessed_images

                pcd_meta = list(self.pairwise(pcd_meta))
                rearranged_pcds = []
                pcds, pcd_mask = pcd_meta[0]
                for i in range(self.img_history_size):
                    pcd, valid = pcds[i], pcd_mask[i]
                    if valid and (math.prod(pcd.shape) > 0) and \
                        (random.random() > self.cond_mask_prob):
                        rearranged_pcds.append((pcd, True))
                    else:
                        rearranged_pcds.append((None, False))

                preprocessed_pcds = []
                pcds_mean = []
                pcds_scale = []
                for pcd, valid in rearranged_pcds:
                    if valid:
                        # Downsample or upsample the point cloud to a fixed number of points
                        processed_pcd = torch.from_numpy(pcd[:, :3].copy()).float()
                        if processed_pcd.shape[0] > self.pcd_max_num:
                            print(f"Processing FPS down sample")
                            o3d_pcd = o3d.geometry.PointCloud()
                            o3d_pcd.points = o3d.utility.Vector3dVector(processed_pcd.numpy())
                    
                            # b. 執行 FPS 降採樣
                            down_sampled_o3d_pcd = o3d_pcd.farthest_point_down_sample(self.pcd_max_num)
                            
                            # c. 將結果轉換回 PyTorch 張量
                            processed_pcd = torch.from_numpy(np.asarray(down_sampled_o3d_pcd.points)).float()
                        elif processed_pcd.shape[0] < self.pcd_min_num:

                            padding_count = self.pcd_min_num - processed_pcd.shape[0]

                            # a. 隨機選擇 `padding_count` 個點作為內插的"種子"
                            rand_indices = torch.randint(processed_pcd.shape[0], size=(padding_count,))
                            seed_points = processed_pcd[rand_indices]

                            # b. 計算所有點之間的成對距離，找到每個種子點的最近鄰 (k=1)
                            dist_matrix = torch.cdist(seed_points, processed_pcd)  # 計算種子點到所有點的距離
                            dist_matrix.fill_diagonal_(float('inf'))  # 忽略點到自身的距離
                            neighbor_indices = torch.argmin(dist_matrix, dim=1)
                            neighbor_points = processed_pcd[neighbor_indices]

                            # c. 在種子點和其最近鄰之間進行隨機線性內插
                            #    生成一個 (padding_count, 1) 的隨機權重
                            weights = torch.rand(padding_count, 1)
                            interpolated_points = seed_points * weights + neighbor_points * (1 - weights)
                            processed_pcd = torch.cat([processed_pcd, interpolated_points], dim=0)
                        
                        # Add noise to the point cloud
                        if self.pcd_noise_std > 0.0:
                            noise = torch.randn_like(processed_pcd) * self.pcd_noise_std
                            processed_pcd += noise
                        pcd_mean = torch.mean(processed_pcd, dim=0)
                        pcds_mean.append(pcd_mean)
                        processed_pcd -= pcd_mean
                        max_dist = torch.max(torch.sqrt(torch.sum(processed_pcd**2, dim=1)))
                        pcds_scale.append(max_dist)
                        processed_pcd /= (max_dist + 1e-6)
                    else:
                        processed_pcd = torch.zeros((self.pcd_max_num, 3), dtype=torch.float32)
                        pcds_mean.append(torch.zeros(3, dtype=torch.float32))
                        pcds_scale.append(torch.tensor(1.0, dtype=torch.float32))

                    pcd = self.pcd_processor(processed_pcd)
                    preprocessed_pcds.append(pcd)
                data_dict["pointclouds"] = preprocessed_pcds
                data_dict["pcd_mean"] = torch.stack(pcds_mean, dim=0)
                data_dict["pcd_scale"] = torch.stack(pcds_scale, dim=0)

                # --- 【!! 新增 !!】 處理關鍵幀 (Keyframes) ---
                
                preprocessed_keyframe_images = []
                preprocessed_keyframe_pcds = []
                keyframe_pcds_mean = []
                keyframe_pcds_scale = []
                
                # 遍歷 key_frame_meta (例如 [kf1_dict, kf2_dict])
                for kf_dict in key_frame_meta: 
                    # for base

                    
                    # --- A. 處理關鍵幀影像 (side_cam, top_cam) ---
                    for cam_key in ["base_cam","top_cam","side_cam"]:
                        image_np = kf_dict.get(cam_key)
                        # 檢查影像是否存在且不為空
                        valid = (image_np is not None and image_np.shape[0] > 0) 
                        
                        if valid:
                            # 假設 HDF5 讀取器已回傳 RGB
                            image = Image.fromarray(image_np) 
                        else:
                            # 如果關鍵幀遺失，使用背景圖
                            image = Image.fromarray(background_image.copy())

                        # (套用與歷史影像相同的增強)
                        if self.image_size is not None:
                            image = transforms.Resize(self.image_size)(image)
                        
                        if valid and self.auto_adjust_image_brightness:
                            img_np_check = np.array(image) # 使用不同變數名稱
                            average_brightness = img_np_check.mean() / 255.0
                            if average_brightness <= 0.15:
                                image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                        
                        if valid and self.image_aug and (random.random() > 0.5):
                            aug_type = random.choice(["corrput_only", "color_only", "both"])
                            if aug_type != "corrput_only":
                                image = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                            if aug_type != "color_only":
                                image = image_corrupt(image)
                        
                        if self.image_aspect_ratio == 'pad':
                            image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                        
                        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                        preprocessed_keyframe_images.append(image_tensor)

                    # --- B. 處理關鍵幀點雲 (pcd) ---
                    # (此邏輯複製並簡化自上方的歷史點雲處理迴圈)
                    pcd_np = kf_dict.get('pcd')
                    # 檢查點雲是否存在且不為空
                    valid = (pcd_np is not None and pcd_np.shape[0] > 0) 
                    
                    if valid:
                        # 假設 HDF5 讀取器已回傳 (N, 3) XYZ
                        # pcd_np[:, :3] 寫法可以同時相容 (N, 3) 和 (N, 6) 輸入
                        processed_pcd_kf = torch.from_numpy(pcd_np[:, :3].copy()).float() # 使用不同變數名稱
                        
                        # (FPS 降採樣)
                        if processed_pcd_kf.shape[0] > self.pcd_max_num:
                            o3d_pcd_kf = o3d.geometry.PointCloud() # 使用不同變數名稱
                            o3d_pcd_kf.points = o3d.utility.Vector3dVector(processed_pcd_kf.numpy())
                            down_sampled_o3d_pcd_kf = o3d_pcd_kf.farthest_point_down_sample(self.pcd_max_num)
                            processed_pcd_kf = torch.from_numpy(np.asarray(down_sampled_o3d_pcd_kf.points)).float()
                        
                        # (插值 增採樣)
                        elif processed_pcd_kf.shape[0] < self.pcd_min_num:
                            padding_count = self.pcd_min_num - processed_pcd_kf.shape[0]
                            rand_indices = torch.randint(processed_pcd_kf.shape[0], size=(padding_count,))
                            seed_points = processed_pcd_kf[rand_indices]
                            dist_matrix = torch.cdist(seed_points, processed_pcd_kf)
                            dist_matrix.fill_diagonal_(float('inf'))
                            neighbor_indices = torch.argmin(dist_matrix, dim=1)
                            neighbor_points = processed_pcd_kf[neighbor_indices]
                            weights = torch.rand(padding_count, 1)
                            interpolated_points = seed_points * weights + neighbor_points * (1 - weights)
                            processed_pcd_kf = torch.cat([processed_pcd_kf, interpolated_points], dim=0)
                        
                        # (加噪 & 正規化)
                        if self.pcd_noise_std > 0.0:
                            noise = torch.randn_like(processed_pcd_kf) * self.pcd_noise_std
                            processed_pcd_kf += noise
                        pcd_mean_kf = torch.mean(processed_pcd_kf, dim=0) # 使用不同變數名稱
                        processed_pcd_kf -= pcd_mean_kf
                        max_dist_kf = torch.max(torch.sqrt(torch.sum(processed_pcd_kf**2, dim=1))) # 使用不同變數名稱
                        pcd_scale_kf = max_dist_kf # 使用不同變數名稱
                        processed_pcd_kf /= (max_dist_kf + 1e-6)
                    
                    else:
                        # (處理無效/空白點雲)
                        processed_pcd_kf = torch.zeros((self.pcd_max_num, 3), dtype=torch.float32)
                        pcd_mean_kf = torch.zeros(3, dtype=torch.float32)
                        pcd_scale_kf = torch.tensor(1.0, dtype=torch.float32)

                    pcd_tensor_kf = self.pcd_processor(processed_pcd_kf) # 使用不同變數名稱
                    
                    preprocessed_keyframe_pcds.append(pcd_tensor_kf)
                    keyframe_pcds_mean.append(pcd_mean_kf)
                    keyframe_pcds_scale.append(pcd_scale_kf)
                
                # 將處理完的關鍵幀資料加入 data_dict
                # (keyframe_images: [KF1_Side, KF1_Top, KF2_Side, KF2_Top])
                data_dict["keyframe_images"] = preprocessed_keyframe_images 
                # (keyframe_pcds: [KF1_PCD, KF2_PCD])
                data_dict["keyframe_pcds"] = preprocessed_keyframe_pcds     
                data_dict["keyframe_pcd_mean"] = torch.stack(keyframe_pcds_mean, dim=0)
                data_dict["keyframe_pcd_scale"] = torch.stack(keyframe_pcds_scale, dim=0)
                # --- 結束新增 ---

                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                for k, v in data_dict.items():
                    assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"
        
                return data_dict
            except BaseException as e:
                # Print the error info
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                # Try incresing the index
                index = (index + 1) % len(self)


class DataCollatorForVAConsumerDataset(object):
    """Collate examples for supervised training."""
    def __init__(self):
        pass

    def __call__(self, instances):
        batch = {
            "states": [],
            "actions": [],
            "state_elem_mask": [],
            "state_norm": [],
            "images": [],
            "pointclouds": [],
            "pcd_mean": [],
            "pcd_scale": [],
            "ctrl_freq": [],
            "keyframe_images": [],  # 加入這行：關鍵幀影像列表
            "keyframe_pcds": [],    # 加入這行：關鍵幀點雲列表
            "keyframe_pcd_mean": [],# 加入這行
            "keyframe_pcd_scale": [], # 加入這行
        }
        
        for instance in instances:
            # Convert all the numpy arrays to tensor
            for key, value in instance.items():
                if key not in ['images', 'pointclouds', 'keyframe_images', 'keyframe_pcds']:
                    if isinstance(value, torch.Tensor):
                        item = value
                    elif isinstance(value, (int, float)):
                        item = torch.tensor(value)
                    else:
                        item = torch.from_numpy(value)
                    batch[key].append(item)
            
            batch["images"].append(torch.stack(instance["images"], dim=0))
            batch['pointclouds'].append(torch.stack(instance['pointclouds'], dim=0))
            batch["keyframe_images"].append(torch.stack(instance["keyframe_images"], dim=0))
            batch['keyframe_pcds'].append(torch.stack(instance['keyframe_pcds'], dim=0))
        
        for key, value_list in batch.items():
            try:
                batch[key] = torch.stack(value_list, dim=0)            
            except Exception as e:
                print(f"[ERROR] key: {key}")
                for i, v in enumerate(value_list):
                    print(f"  entry {i}: shape={v.shape}")
                raise e
        return batch
