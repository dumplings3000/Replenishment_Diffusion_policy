python -m scripts.agilex_inference \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/your_finetuned_ckpt.pt" \  
    --ctrl_freq=15    # your control frequency
