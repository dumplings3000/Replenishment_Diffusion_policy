import h5py
import os

# 🚨 根據你的數據路徑修改這個檔案名
H5_FILE_PATH = "/home/user/RDP/data/finetune/trajectory.h5"

GROUP_TO_DELETE = 'traj_9'

if not os.path.exists(H5_FILE_PATH):
    print(f"錯誤：檔案 {H5_FILE_PATH} 不存在。請檢查路徑是否正確。")
else:
    try:
        # 以追加模式 ('a') 打開檔案，允許讀寫
        with h5py.File(H5_FILE_PATH, 'a') as f:
            
            print(f"--- 檢查檔案：{H5_FILE_PATH} ---")
            
            if GROUP_TO_DELETE in f:
                print(f"✅ 找到群組 /{GROUP_TO_DELETE}。正在刪除...")
                
                # 使用 del 語法來刪除 Group 及其所有內容
                del f[GROUP_TO_DELETE]
                
                print(f"🎉 刪除完成。")
                print(f"當前檔案的根群組：{list(f.keys())}")
            else:
                print(f"⚠️ 警告：群組 /{GROUP_TO_DELETE} 不存在於檔案中。無需刪除。")
                print(f"當前檔案的根群組：{list(f.keys())}")
                
    except Exception as e:
        print(f"🚨 處理 HDF5 檔案時發生錯誤: {e}")