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
                            _len = f[f'{traj_key}/actions'].shape[0] + 1
                            self.episodes.append((file_path, traj_key))
                            episode_lens.append(_len)
                        except KeyError:
                            print(f"警告：軌跡 {traj_key} in {file_path} 缺少 'actions'數據，已跳過。")
            except Exception as e:
                print(f"警告：無法讀取 {file_path}: {e}")
        print(f"共有 {len(file_paths)} 個檔案，共 {len(self.episodes)} 個軌跡。")
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
    
    def get_item(self, index=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

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
                valid, sample = self.parse_hdf5_file_state_only(f, traj_key)if state_only else self.parse_hdf5_file(f, traj_key)    
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
            actions = f[f'{traj_key}/actions'][:]
            num_steps = qpos.shape[0] - 1
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
                print(f"警告：軌跡 {traj_key} in {file_path} 沒有有效移動，已跳過。")
                return False, None

            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)
            assert step_id >= 0 and step_id < num_steps, f"step_id: {step_id}, num_steps: {num_steps}"

            meta = {
                "#steps": num_steps,
                "step_id": step_id,
            }

            if actions.shape[1] == 8:
                # Rescale gripper to [0, 1]
                qpos = qpos / np.array(
                [[1, 1, 1, 1, 1, 1, 1, 0.04, 0.04]] 
                )
                qpos = np.concatenate(
                        (qpos[:, :7], np.mean(qpos[:, -2:], axis=1, keepdims=True)), 
                        axis=1
                    )
                state = qpos
                # Fill the state/action into the unified vector
                def fill_in_state(values):
                    # Target indices corresponding to your state space
                    # In this example: 6 joints + 1 gripper for each arm
                    UNI_STATE_INDICES = [
                        STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)
                    ] + [
                        STATE_VEC_IDX_MAPPING["gripper_open"]
                    ]
                    # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                    uni_vec[..., UNI_STATE_INDICES] = values
                    return uni_vec
            elif actions.shape[1] == 7:
                tcp_pos = f[f'{traj_key}/obs/extra/tcp_pose'][:]
                state = tcp_pos
                def fill_in_state(values):
                    # Target indices corresponding to your state space
                    # In this example: 6 joints + 1 gripper for each arm
                    UNI_STATE_INDICES = [
                        STATE_VEC_IDX_MAPPING["eef_angle_0"]
                    ] + [
                        STATE_VEC_IDX_MAPPING["eef_angle_1"]
                    ] + [
                        STATE_VEC_IDX_MAPPING["eef_angle_2"]
                    ] + [
                        STATE_VEC_IDX_MAPPING["eef_angle_3"]
                    ] + [
                        STATE_VEC_IDX_MAPPING["eef_angle_4"]
                    ] + [
                        STATE_VEC_IDX_MAPPING["eef_angle_5"]
                    ] + [
                        STATE_VEC_IDX_MAPPING["gripper_open"]
                    ]
                    # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                    uni_vec[..., UNI_STATE_INDICES] = values
                    return uni_vec
            else:
                return False, None
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
                return imgs
            base_camera = parse_img('base_camera')
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            base_camera_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            hand_camera = parse_img("hand_camera")
            hand_camera_mask = base_camera_mask.copy()
            side_camera = parse_img("side_camera")
            side_camera_mask = base_camera_mask.copy()

            def parse_pcd():
                pcd_path = f'{traj_key}/obs/pointcloud/xyzw'
                if pcd_path not in f: return np.zeros((self.IMG_HISORY_SIZE, 0, 0), dtype=np.float32)
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
                "pointcloud_mask": pcd_mask
            }
        except Exception as e:
            print(f"警告：無法在 parse 階段讀取 {f.filename} {traj_key}: {e}")
            return False, None

    def parse_hdf5_file_state_only(self, f, traj_key):
        """
        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        qpos = f[f'{traj_key}/obs/agent/qpos'][:]
        actions = f[f'{traj_key}/actions'][:]
        # num_steps = qpos.shape[0]
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
            print(f"警告：軌跡 {traj_key} in {file_path} 沒有有效移動，已跳過。")
            return False, None


        if actions.shape[1] == 8:
            # Rescale gripper to [0, 1]
            qpos = qpos / np.array(
            [[1, 1, 1, 1, 1, 1, 1, 0.04, 0.04]] 
            )
            qpos = np.concatenate(
                    (qpos[:, :7], np.mean(qpos[:, -2:], axis=1, keepdims=True)), 
                    axis=1
                )
            state = qpos[first_idx-1:]
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)
                ] + [
                    STATE_VEC_IDX_MAPPING["gripper_open"]
                ]
                # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
        elif actions.shape[1] == 7:
            tcp_pos = f[f'{traj_key}/obs/extra/tcp_pose'][:]
            state = tcp_pos[first_idx-1:]
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING["eef_angle_0"]
                ] + [
                    STATE_VEC_IDX_MAPPING["eef_angle_1"]
                ] + [
                    STATE_VEC_IDX_MAPPING["eef_angle_2"]
                ] + [
                    STATE_VEC_IDX_MAPPING["eef_angle_3"]
                ] + [
                    STATE_VEC_IDX_MAPPING["eef_angle_4"]
                ] + [
                    STATE_VEC_IDX_MAPPING["eef_angle_5"]
                ] + [
                    STATE_VEC_IDX_MAPPING["gripper_open"]
                ]
                # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
        action = actions[first_idx-1:]
        state = fill_in_state(state)
        action = fill_in_state(action)

        
        # Return the resulting sample
        return True, {
            "state": state,
            "action": action
        }

if __name__ == "__main__":
    ds = HDF5VADataset("data/pretrain/")
    for i in range(len(ds)):
        print(f"Processing episode {i+1}/{len(ds)}...")
        ds.get_item(i)
