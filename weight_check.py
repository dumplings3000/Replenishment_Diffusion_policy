import torch
import argparse

def list_weights(weight_path, show_values=False):
    print(f"📦 Loaded weight file: {weight_path}")
    model_data = torch.load(weight_path, map_location='cpu')

    # 若是典型 pytorch checkpoint，權重通常在 'state_dict' 或 'model' 等 key 裡
    if isinstance(model_data, dict):
        # 先找一個可能的 key，常見 state dict 在這些裡面
        for candidate_key in ['state_dict', 'model', 'module']:
            if candidate_key in model_data:
                print(f"🔍 Found '{candidate_key}' key in checkpoint, using it as state dict.")
                model_data = model_data[candidate_key]
                break

    # 遞迴列印 tensor 名稱、形狀、dtype
    def recurse_print(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                recurse_print(v, prefix + k + ".")
        elif isinstance(d, torch.Tensor):
            print(f"{prefix[:-1]}: shape={tuple(d.shape)} dtype={d.dtype}")
            if show_values:
                print(d)

    print("📚 Listing parameters:")
    recurse_print(model_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List all weights in a PyTorch checkpoint")
    parser.add_argument("weight_path", type=str, help="Path to the weight file (e.g. .pt)")
    parser.add_argument("--show_values", action="store_true", help="Show tensor values")
    args = parser.parse_args()

    list_weights(args.weight_path, args.show_values)
