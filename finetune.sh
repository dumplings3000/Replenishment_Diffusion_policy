export VISION_ENCODER_NAME="facebook/dinov2-large"
export PCD_ENCODER_NAME="models/ULIP/ckpt/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt"
export OUTPUT_DIR="/media/lab901-server/DEE1-00E6/checkpoints/RDP_SECOND_train"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/home/user/cutlass"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export CUDA_LAUNCH_BLOCKING=1

export WANDB_PROJECT="REPLENISHMENT_DIFFUSION_POLICY second training new"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

deepspeed --hostfile=hostfile.txt main.py \
    --deepspeed="./configs/zero1.json" \
    --model_path="models/rdp/ckpt/pytorch_model.bin"\
    --pretrain_vision_encoder=$VISION_ENCODER_NAME \
    --pretrain_pointcloud_encoder=$PCD_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=2 \
    --sample_batch_size=2 \
    --max_train_steps=200000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=2000 \
    --lr_num_cycles=1 \
    --lr_power=1.0 \
    --learning_rate=1e-5 \
    --gradient_accumulation_steps=16 \
    --dataloader_num_workers=12 \
    --state_noise_snr=20 \
    --max_grad_norm=0.5 \
    --report_to=wandb \
    --adam_epsilon=1e-4 \
    --allow_tf32 \
    --mixed_precision="bf16" \
    --adam_weight_decay=0.05 
    # --resume_from_checkpoint="latest"

    # Use this to resume training from some previous checkpoint
