import os
import fnmatch
import h5py
import yaml
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING
import threading


class HDF5VADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self, dataset_pth):
        self.episodes = []
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['history_size']
        self.STATE_DIM = config['common']['state_dim']

        file_paths = []
        for root, _, files in os.walk(dataset_pth):
            for filename in fnmatch.filter(files, '*.h5'):
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
                
        # Get each episode's len
        episode_lens = []
        for file_path in file_paths:
            try:
                with h5py.File(file_path, 'r') as f:
                    for traj_key in f.keys():
                        if not traj_key.startswith('traj_'):
                            continue # 跳過非軌跡群組
                        try:
                            _len = f[f'{traj_key}/actions/joint_action'].shape[0]
                            self.episodes.append((file_path, traj_key))
                            episode_lens.append(_len)
                        except KeyError:
                            print(f"警告：軌跡 {traj_key} in {file_path} 缺少 'actions/joint_action'數據，已跳過。")
            except Exception as e:
                print(f"警告：無法讀取 {file_path}: {e}")
        # print(f"共有 {len(file_paths)} 個檔案，共 {len(self.episodes)} 個軌跡。")
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

        self._local_storage = threading.local()
        
    def _get_file_handle(self, file_path):
        """為每個 worker 惰性地打開和管理檔案控制代碼"""
        # 如果當前 worker 的儲存物件還不存在，則初始化
        if not hasattr(self._local_storage, 'handles'):
            self._local_storage.handles = {}

        # 如果該檔案的 handle 還沒被這個 worker 打開過，就打開它
        if file_path not in self._local_storage.handles:
            self._local_storage.handles[file_path] = h5py.File(file_path, 'r', swmr=True)
        
        return self._local_storage.handles[file_path]
    
    def __len__(self):
        return len(self.episodes)
    
    def get_item(self, index=None):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        if not self.episodes:
            raise RuntimeError("數據集中沒有找到任何有效的軌跡。")
        
        while True:
            if index is None:
                episode_idx = np.random.choice(len(self.episodes), p=self.episode_sample_weights)
            else:
                episode_idx = index
            file_path, traj_key = self.episodes[episode_idx]
            try:
                f = self._get_file_handle(file_path)
                valid, sample = self.parse_hdf5_file(f, traj_key)    
                if valid: return sample
            except Exception as e:
                # 如果依然出錯，列印更詳細的資訊
                print(f"Worker {os.getpid()} 在處理 {file_path} 的 {traj_key} 時發生嚴重錯誤: {e}")
                # 清理掉可能有問題的 handle，讓下次重新開啟
                if hasattr(self._local_storage, 'handles') and file_path in self._local_storage.handles:
                    self._local_storage.handles[file_path].close()
                    del self._local_storage.handles[file_path]
            print(f"警告: 軌跡 {traj_key} in {file_path} 無效，重新採樣...")
            index = None
    
    def parse_hdf5_file(self, f, traj_key):
        try:
            qpos = f[f'{traj_key}/obs/agent/qpos'][:]
            actions = f[f'{traj_key}/actions/joint_action'][:]
            num_steps = qpos.shape[0]
            # if num_steps < 128:
            #     return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                print(f"警告：軌跡 {traj_key} 沒有有效移動，已跳過。")
                return False, None

            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)
            assert step_id >= 0 and step_id < num_steps, f"step_id: {step_id}, num_steps: {num_steps}"

            meta = {
                "#steps": num_steps,
                "step_id": step_id,
            }

            
            state = qpos
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["gripper_open"]
                ]
                # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state_std = np.std(state, axis=0)
            state_mean = np.mean(state, axis=0)
            state_norm = np.sqrt(np.mean(state**2, axis=0))
            state = state[step_id:step_id+1]
            action = actions[step_id:step_id+self.CHUNK_SIZE]
            if action.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                action = np.concatenate([
                    action,
                    np.tile(action[-1:], (self.CHUNK_SIZE-action.shape[0], 1))
                ], axis=0)
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            action = fill_in_state(action)
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            
            # Parse the images
            def parse_img(key):
                img_path = f'{traj_key}/obs/sensor_data/{key}/rgb'
                if img_path not in f: return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
                # imgs = []
                # for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                #     img = f[img_path][i]
                #     imgs.append(img)
                # imgs = np.stack(imgs)
                # print(f"{key}_imgs_shape: {imgs.shape}")
                start_idx = max(0, step_id - self.IMG_HISORY_SIZE + 1)
                end_idx = step_id + 1
                
                # 2. 一次性將整個歷史區塊讀取到 NumPy 陣列中
                imgs = f[img_path][start_idx:end_idx]
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                imgs = imgs[..., ::-1].copy()
                return imgs
            base_camera = parse_img('base_camera')
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            base_camera_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            hand_camera = parse_img("top_camera")
            hand_camera_mask = base_camera_mask.copy()
            side_camera = parse_img("side_camera")
            side_camera_mask = base_camera_mask.copy()

            def parse_pcd():
                pcd_path = f'{traj_key}/obs/pointcloud/xyz'
                if pcd_path not in f: return np.zeros((self.IMG_HISORY_SIZE, 0, 3), dtype=np.float32)
                # pcds = []
                # for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                #     pcd = f[pcd_path][i]
                #     pcds.append(pcd)
                # pcds = np.stack(pcds)
                start_idx = max(0, step_id - self.IMG_HISORY_SIZE + 1)
                end_idx = step_id + 1
                
                # 一次性讀取所有需要的點雲
                pcds = f[pcd_path][start_idx:end_idx]
                if pcds.shape[0] < self.IMG_HISORY_SIZE:
                    #Pad the pcds using the first pcd
                    pcds = np.concatenate([
                        np.tile(pcds[:1], (self.IMG_HISORY_SIZE-pcds.shape[0], 1, 1)),
                        pcds
                    ], axis=0)
                return pcds
            pcd = parse_pcd()
            pcd_mask = base_camera_mask.copy()

            def get_img_at_idx(key, idx):
                img_path = f'{traj_key}/obs/sensor_data/{key}/rgb'
                if img_path not in f:
                    print(f"警告: 找不到 {img_path} (for keyframe)")
                    # 回傳 (H,W,C) = (0,0,3)，一個空的影像
                    return np.zeros((0, 0, 3), dtype=np.uint8) 
                try:
                    img_bgr = f[img_path][idx]
                    img_rgb = img_bgr[..., ::-1].copy()
                    return img_rgb
                except Exception as e:
                    print(f"警告: 讀取 keyframe {img_path} at {idx} 失敗: {e}")
                    return np.zeros((0, 0, 3), dtype=np.uint8)
                
            def get_pcd_at_idx(idx):
                pcd_path = f'{traj_key}/obs/pointcloud/xyz'
                if pcd_path not in f:
                    print(f"警告: 找不到 {pcd_path} (for keyframe)")
                     # 回傳 (N, C) = (0, 3)，一個空的點雲
                    return np.zeros((0, 3), dtype=np.float32)
                try:
                    return f[pcd_path][idx]
                except Exception as e:
                    print(f"警告: 讀取 keyframe {pcd_path} at {idx} 失敗: {e}")
                    return np.zeros((0, 3), dtype=np.float32)
                
            key_frames_data = []
                
            try:
                keyframe_indices = f[f'{traj_key}/keyframe_indices'][:]
                kf_idx_1 = keyframe_indices[0]
                kf_idx_2 = keyframe_indices[1]
                if step_id > kf_idx_1:
                    # 讀取第一組關鍵幀 (side, top, pcd)
                    keyframe_1 = {
                        "side_cam": get_img_at_idx("side_camera", kf_idx_1),
                        "top_cam": get_img_at_idx("top_camera", kf_idx_1),
                        "base_cam": None,
                        "pcd": get_pcd_at_idx(kf_idx_1),
                    }
                else:
                    keyframe_1 = {
                        "side_cam": None,
                        "top_cam": None,
                        "base_cam": None,
                        "pcd": None,
                    }
                key_frames_data.append(keyframe_1)

                if step_id > kf_idx_2:
                    # 讀取第二組關鍵幀 (side, top, pcd)
                    keyframe_2 = {
                        "side_cam": get_img_at_idx("side_camera", kf_idx_2),
                        "top_cam": get_img_at_idx("top_camera", kf_idx_2),
                        "base_cam": None,
                        "pcd": get_pcd_at_idx(kf_idx_2),
                    }
                else:
                    keyframe_2 = {
                        "side_cam": None,
                        "top_cam": None,
                        "base_cam": None,
                        "pcd": None,
                    }
                key_frames_data.append(keyframe_2)

            except KeyError:
                print(f"警告: 找不到 {traj_key}/keyframe_indices。將回傳空的關鍵幀。")
                # 如果沒有 keyframe_indices，就回傳兩組空的 dict
                empty_kf_dict = {
                    "side_cam": np.zeros((0, 0, 3), dtype=np.uint8),
                    "top_cam": np.zeros((0, 0, 3), dtype=np.uint8),
                    "pcd": np.zeros((0, 3), dtype=np.float32),
                }
                key_frames_data.append(empty_kf_dict)
                # .copy() 確保它們是兩個獨立的字典
                key_frames_data.append(empty_kf_dict.copy())



            # print(f"state_shape: {state.shape}")
            # print(f"state_std_shape: {state_std.shape}")
            # print(f"state_mean_shape: {state_mean.shape}")
            # print(f"state_norm_shape: {state_norm.shape}")
            # print(f"action_shape: {action.shape}")
            # print(f"state_indicator_shape: {state_indicator.shape}")
            # print(f"base_camera_shape: {base_camera.shape}")
            # print(f"base_camera_mask_shape: {base_camera_mask.shape}")
            # print(f"hand_camera_shape: {hand_camera.shape}")
            # print(f"hand_camera_mask_shape: {hand_camera_mask.shape}")
            # print(f"side_camera_shape: {side_camera.shape}")
            # print(f"side_camera_mask_shape: {side_camera_mask.shape}")
            # print(f"pcd_shape: {pcd.shape}")
            # print(f"pcd_mask_shape: {pcd_mask.shape}")

            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": action,
                "state_indicator": state_indicator,
                "base_cam": base_camera,
                "base_cam_mask": base_camera_mask,
                "hand_cam": hand_camera,
                "hand_cam_mask": hand_camera_mask,
                "side_cam": side_camera,
                "side_cam_mask": side_camera_mask,
                "pointcloud": pcd,
                "pointcloud_mask": pcd_mask,
                "key_frames_data": key_frames_data
            }
        except Exception as e:
            print(f"警告：無法在 parse 階段讀取 {f.filename} {traj_key}: {e}")
            return False, None

if __name__ == "__main__":
    import open3d as o3d
    import cv2
    
    def show_pcd(pcd_data, window_name="Point Cloud"):
        if not pcd_data.size > 0:
            print("  [PCD: 數據為空，跳過顯示]")
            return
            
        pcd = o3d.geometry.PointCloud()
        
        points = pcd_data # 【!! 更改 !!】 pcd_data 現在就是 (N, 3)
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # (除錯 XYZ 座標)
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        print(f"  [DEBUG PCD Coords] X range: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  [DEBUG PCD Coords] Y range: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  [DEBUG PCD Coords] Z range: [{z_min:.3f}, {z_max:.3f}]")
        
        # 2. 根據 Z 軸高度上色 (僅用 Numpy)
        if (z_max - z_min) < 1e-4:
            print("  [PCD AutoColor] Z 軸範圍太小 (點雲是平的), 設為灰色。")
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        else:
            print("  [PCD AutoColor] 正在根據 Z 軸 (高度) 自動上色 (藍 -> 紅)...")
            normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
            colors = np.zeros((len(points), 3))
            colors[:, 0] = normalized_z
            colors[:, 2] = 1.0 - normalized_z
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 3. 設置 Open3D 渲染器 (不變)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        render_option = vis.get_render_option()
        render_option.point_color_option = o3d.visualization.PointColorOption.Color
        render_option.light_on = True 
        print(f"  [PCD: 正在顯示 {len(points)} 個點... (請關閉 3D 視窗以繼續)]")
        vis.add_geometry(pcd)
        vis.run() 
        vis.destroy_window()

    # --- 主執行程式 (依序顯示所有軌跡) ---
    
    ds = HDF5VADataset("data/finetune/") 
    if len(ds) == 0:
        print("!! 錯誤: 數據集中沒有找到任何軌跡 !!")
    else:
        print(f"數據集加載成功，共有 {len(ds)} 個軌跡。")
        print("--- 開始依序顯示所有軌跡的關鍵幀 ---")

        for i in range(len(ds)):
            print(f"\n=======================================================")
            print(f"--- 正在處理軌跡 {i+1} / {len(ds)} (index {i}) ---")
            print(f"=======================================================")
            try:
                sample = ds.get_item(i) 
                print("樣本讀取成功！")
                if "key_frames_data" in sample:
                    key_frames = sample['key_frames_data']
                    print(f"成功找到 'key_frames_data'，共 {len(key_frames)} 組關鍵幀。")
                    for kf_idx, kf in enumerate(key_frames):
                        print(f"\n--- 正在顯示第 {kf_idx+1} 組關鍵幀 ---")
                        side_img_rgb = kf.get('side_cam')
                        top_img_rgb = kf.get('top_cam')
                        valid_images_rgb = []
                        img_names = []
                        if side_img_rgb is not None and side_img_rgb.size > 0:
                            valid_images_rgb.append(side_img_rgb)
                            img_names.append("Side")
                        if top_img_rgb is not None and top_img_rgb.size > 0:
                            valid_images_rgb.append(top_img_rgb)
                            img_names.append("Top")
                        if valid_images_rgb:
                            combined_img_rgb = np.hstack(valid_images_rgb)
                            window_title = f"Traj {i+1}, KF {kf_idx+1} ({' | '.join(img_names)})"
                            print(f"  [IMG: 顯示 {img_names} 影像...]")
                            print(f"  [IMG: 在影像視窗上按任意鍵以顯示點雲...]")
                            # 直接顯示 RGB 影像 (顏色在 cv2 視窗中會不對)
                            cv2.imshow(window_title, combined_img_rgb) 
                            cv2.waitKey(0) 
                            cv2.destroyWindow(window_title)
                        else:
                            print("  [IMG: 此關鍵幀無有效影像]")
                        pcd_data = kf.get('pcd')
                        if pcd_data is not None and pcd_data.size > 0:
                            show_pcd(pcd_data, window_name=f"Traj {i+1}, KF {kf_idx+1} PCD")
                        else:
                            print("  [PCD: 此關鍵幀無有效點雲]")
                    print(f"\n--- 軌跡 {i+1} 關鍵幀顯示完畢 ---")
                else:
                    print("!! 錯誤: 樣本中未找到 'key_frames_data' !!")
            except Exception as e:
                print(f"!! 處理軌跡 {i+1} (index {i}) 時發生嚴重錯誤: {e} !!")
                print("!! 將跳過此軌跡繼續 !!")
                import traceback
                traceback.print_exc()
        print("\n=======================================================")
        print("--- 所有軌跡測試完畢 ---")
        cv2.destroyAllWindows()
# if __name__ == "__main__":
#     ds = HDF5VADataset("data/finetune/")
    # for i in range(len(ds)):
    #     print(f"Processing episode {i+1}/{len(ds)}...")
    #     ds.get_item(i)
