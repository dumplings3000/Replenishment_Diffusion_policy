import h5py
import numpy as np
import os
import tqdm

SOURCE_FILE = 'data/pretrain/trajectories.h5'
# 建立一個新的資料夾來存放拆分後的檔案
OUTPUT_DIR = 'data/pretrain/trajectories_split' 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"已建立輸出資料夾: {OUTPUT_DIR}")

print(f"開始從 {SOURCE_FILE} 拆分資料到 {OUTPUT_DIR}...")

with h5py.File(SOURCE_FILE, 'r') as f_src:
    traj_keys = list(f_src.keys())
    for key in tqdm.tqdm(traj_keys):
        if not key.startswith('traj_'):
            continue
        
        try:
            # 將一個軌跡中的所有相關資料讀取出來
            traj_group = f_src[key]
            
            # 建立一個字典來儲存所有資料陣列
            data_to_save = {}
            for dataset_name in traj_group.keys():
                # 例如 'actions', 'obs', 'extra' 等
                # 如果是群組，就遍歷群組內的資料集
                if isinstance(traj_group[dataset_name], h5py.Group):
                    for sub_dataset_name in traj_group[dataset_name].keys():
                        # ...可以根據您的需求決定要儲存哪些資料
                        pass # 這裡為了簡化，先跳過巢狀群組
                else:
                    data_to_save[dataset_name] = traj_group[dataset_name][()]

            # 特別處理 obs 裡的內容
            data_to_save['qpos'] = traj_group['obs/agent/qpos'][:]
            # 根據您的程式碼，您可能還需要 'obs/extra/tcp_pose' 等
            if 'obs/extra/tcp_pose' in traj_group:
                 data_to_save['tcp_pose'] = traj_group['obs/extra/tcp_pose'][:]
            
            # 處理影像和點雲 (如果需要且記憶體允許)
            # 警告：如果影像和點雲很大，每個 .npz 檔案也會很大
            # for cam in ['base_camera', 'hand_camera', 'side_camera']:
            #     img_path = f'obs/sensor_data/{cam}/rgb'
            #     if img_path in traj_group:
            #         data_to_save[f'{cam}_rgb'] = traj_group[img_path][:]

            # 定義輸出的 .npz 檔案路徑
            output_path = os.path.join(OUTPUT_DIR, f"{key}.npz")
            # 使用 numpy.savez_compressed 來儲存成壓縮的 .npz 檔案
            np.savez_compressed(output_path, **data_to_save)

        except Exception as e:
            print(f"處理軌跡 {key} 時發生錯誤，已跳過。錯誤: {e}")

print("\n資料拆分完畢！")