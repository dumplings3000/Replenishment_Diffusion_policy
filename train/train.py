#!/usr/bin/env python
# coding=utf-8
import os
import math
import yaml
import copy
import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
from safetensors.torch import load_model

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed

import transformers

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available


from models.ema_model import EMAModel
from models.multimodal_encoder.DINOv2_encoder import VisionEncoder
from models.multimodal_encoder.PCD_encoder import PointTower
from models.rdp_runner import model_Runner
from train.dataset import DataCollatorForVAConsumerDataset, VAConsumerDataset
from train.sample import log_sample_res


if is_wandb_available():
    import wandb

def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # Set up logging
    logging_dir = Path(args.output_dir, args.logging_dir)
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Set up accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
        total_limit=args.checkpoints_total_limit
    )

    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed before initializing model.
    if args.seed is not None:
        set_seed(args.seed)

    torch.autograd.set_detect_anomaly(True)

    # create output directory if needed
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # set the weights dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Enable TF32 for faster training on Ampere GPUs(GTX30 up or A100)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Use 8-bit Adam Optimizer for lower memory (16GB) usage
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Scale learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    """
    ==========================================Model creation, loading, and EMA model creation===========================================
    """
    # # Encoder creation
    vision_encoder = VisionEncoder(vision_tower=args.pretrain_vision_encoder)
    image_processor = vision_encoder.image_processor
    pcd_encoder = PointTower(args.pretrain_pointcloud_encoder)
    pcd_processor = pcd_encoder.pcd_processor

   
    # Model bacbone creation
    logger.info("Constructing model from provided config.")
    # Calculate the image condition length
    vision_cond_len = (config["common"]["num_cameras"] * (vision_encoder.num_patches+2) + pcd_encoder.num_patches +2)
    global_cond_len = config["model"]["global_cond_token_len"]
    img_cond_len = vision_encoder.num_patches
    pcd_cond_len = pcd_encoder.num_patches
    rdp = model_Runner(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        img_token_dim=config["model"]["img_token_dim"],
        pcd_token_dim=config["model"]["pcd_token_dim"],
        global_pcd_dim=config["model"]["pcd_embed_dim"],
        vision_token_dim=config["model"]["vision_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        img_token_len=img_cond_len,
        pcd_token_len=pcd_cond_len,
        vision_cond_len=vision_cond_len,
        # history_cond_len=global_cond_len,
        vision_pos_embed_config=[
            ("vision", vision_cond_len),  
        ],
        # global_pos_embed_config=[
        #     ("global", 1344),
        # ],
        num_cam=config["common"]["num_cameras"],
        dtype=weight_dtype,
    )

    rdp.load_model(args.model_path)

    # logger.info("Compiling model with torch.compile...")
    # rdp = torch.compile(rdp, mode="reduce-overhead")
    
    # EMA model, Use to update the model weights with exponential moving average, It's can help to improve the performance of the model.                                      
    ema_rdp = copy.deepcopy(rdp)
    ema_model = EMAModel(ema_rdp, 
                        update_after_step=config["model"]["ema"]["update_after_step"],
                        inv_gamma=config["model"]["ema"]["inv_gamma"],
                        power=config["model"]["ema"]["power"],
                        min_value=config["model"]["ema"]["min_value"],
                        max_value=config["model"]["ema"]["max_value"])
    """
    ==============================================================================================================================
    """

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdp))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    # Optimizer creation
    LR_HIGH = args.learning_rate   # Group 3 (新模組, "沒在權重檔裡的")
    LR_MID = args.learning_rate * 0.5  # Group 2 (其他權重, "權重檔裡的沒被凍的")
    LR_LOW = args.learning_rate * 0.1  # Group 1 (核心權重, "原本被凍起來的")
    try:
        optimizer_grouped_parameters = rdp.get_parameter_groups(
            lr_low=LR_LOW,
            lr_mid=LR_MID,
            lr_high=LR_HIGH
        )
    except AttributeError:
        # 處理 accelerate DDP 包裝
        optimizer_grouped_parameters = rdp.module.get_parameter_groups(
            lr_low=LR_LOW,
            lr_mid=LR_MID,
            lr_high=LR_HIGH
        )
    
    # params_to_optimize = rdp.parameters()
    optimizer = optimizer_class(
        optimizer_grouped_parameters,
        # lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler 
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    """
    ===========================================DATA Loader, batching, and collating====================================
    """
    
    # # Dataset and DataLoaders creation:                                                           
    train_dataset = VAConsumerDataset(
        config=config["dataset"],
        image_processor=image_processor,
        pcd_processor=pcd_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["history_size"],
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
    )
    sample_dataset = VAConsumerDataset(
        config=config["dataset"],
        image_processor=image_processor,
        pcd_processor=pcd_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["history_size"],
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
    )                              
    
    data_collator = DataCollatorForVAConsumerDataset()                                                        
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    """
    ======================================================================================================================
    """         
    # # Prepare everything with accelerator.
    rdp, optimizer, train_dataloader,sample_dataloader, lr_scheduler = accelerator.prepare(
        rdp, optimizer, train_dataloader, sample_dataloader, lr_scheduler                   
    )
    
    # # =============================================================================
    # #           模型狀態驗證 (請將這段程式碼加在 prepare 之後)
    # # =============================================================================
    accelerator.print("\n" + "="*60)
    accelerator.print("VERIFYING MODEL AND OPTIMIZER STATE (POST-PREPARE)")
    accelerator.print("="*60)

    unwrapped_model = accelerator.unwrap_model(rdp)

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in unwrapped_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    accelerator.print(f"Total Parameters:     {total_params / 1_000_000:.2f} M")
    accelerator.print(f"Trainable Parameters: {trainable_params / 1_000_000:.2f} M")
    accelerator.print(f"Frozen Parameters:    {frozen_params / 1_000_000:.2f} M")
    accelerator.print("-" * 60)

    # =======================================================================
    # =======================================================================
    accelerator.print("VERIFYING OPTIMIZER STATE (3 GROUPS)")
    
    # 檢查 optimizer 是否真的被 accelerator.prepare 包裝了
    accelerator.print(f"Optimizer class: {type(optimizer)}")

    optimizer_total_params = 0
    
    if not hasattr(optimizer, 'param_groups'):
        accelerator.print("[ERROR] Optimizer object has no 'param_groups'. Something is wrong.")
    else:
        accelerator.print(f"Found {len(optimizer.param_groups)} parameter groups in optimizer.")
        # 迭代檢查 accelerator 處理過的 optimizer 中的每一個 group
        for i, group in enumerate(optimizer.param_groups):
            group_params_count = sum(p.numel() for p in group['params'])
            optimizer_total_params += group_params_count
            # ！！這裡會印出 accelerator 真正使用的學習率！！
            accelerator.print(f"  - Optimizer Group {i} (LR: {group['lr']}): {group_params_count / 1_000_000:.2f} M params")

    accelerator.print(f"Total Parameters in Optimizer: {optimizer_total_params / 1_000_000:.2f} M")

    if trainable_params == optimizer_total_params:
        accelerator.print("[OK] Optimizer parameters match trainable model parameters.")
    else:
        accelerator.print("[!! WARNING !!] Mismatch between optimizer and model trainable parameters.")
        accelerator.print(f"  Trainable model params: {trainable_params}")
        accelerator.print(f"  Optimizer total params: {optimizer_total_params}")

    accelerator.print("="*60 + "\n")
    # accelerator.print("\n" + "="*60)
    # accelerator.print("VERIFYING MODEL AND OPTIMIZER STATE")
    # accelerator.print("="*60)

    # unwrapped_model = accelerator.unwrap_model(rdp)

    # total_params = 0
    # trainable_params = 0
    # frozen_params = 0

    # trainable_param_names = []
    # frozen_param_names = []

    # for name, param in unwrapped_model.named_parameters():
    #     total_params += param.numel()
    #     if param.requires_grad:
    #         trainable_params += param.numel()
    #         trainable_param_names.append(name)
    #     else:
    #         frozen_params += param.numel()
    #         frozen_param_names.append(name)

    # accelerator.print(f"Total Parameters:     {total_params / 1_000_000:.2f} M")
    # accelerator.print(f"Trainable Parameters: {trainable_params / 1_000_000:.2f} M")
    # accelerator.print(f"Frozen Parameters:    {frozen_params / 1_000_000:.2f} M")
    # accelerator.print("-" * 60)

    # accelerator.print(f"Found {len(trainable_param_names)} trainable parameter tensors.")
    # if trainable_param_names:
    #     accelerator.print("Examples of trainable parameters:")
    #     for name in trainable_param_names[:5]:
    #         accelerator.print(f"  - {name}")

    # accelerator.print("-" * 60)
    # accelerator.print(f"Found {len(frozen_param_names)} frozen parameter tensors.")
    # if frozen_param_names:
    #     accelerator.print("Examples of frozen parameters:")
    #     for name in frozen_param_names[:5]:
    #         accelerator.print(f"  - {name}")
    # accelerator.print("-" * 60)


    # # 3. 關鍵檢查：確認優化器中的參數數量是否與模型中可訓練的參數數量一致
    # #    這是非常常見的錯誤來源！
    # optimizer_total_params = sum(p.numel() for p in optimizer.param_groups[0]['params'])

    # accelerator.print("VERIFYING OPTIMIZER STATE")
    # accelerator.print(f"Parameters in Optimizer: {optimizer_total_params / 1_000_000:.2f} M")

    # if trainable_params == optimizer_total_params:
    #     accelerator.print("[OK] Optimizer parameters match trainable model parameters.")
    # else:
    #     accelerator.print("[!! WARNING !!] Mismatch between optimizer and model trainable parameters.")
    #     accelerator.print("This can happen if you freeze layers AFTER initializing the optimizer.")
    #     accelerator.print("Please ensure you filter parameters BEFORE passing them to the optimizer.")

    # accelerator.print("="*60 + "\n")
    # # =============================================================================
    # #                      驗證結束
    # # =============================================================================

    ema_rdp.to(accelerator.device, dtype=weight_dtype)
    vision_encoder.to(accelerator.device, dtype=weight_dtype)
    pcd_encoder.to(accelerator.device, dtype=torch.float32)

    # math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    """
    =====================================================Training===========================================================
    """
     # initialize the trackers we use
    if accelerator.is_main_process:
        accelerator.init_trackers("Replenishment Diffusion Policy", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # Load from a pretrained checkpoint
    if (                                        #check the condition if we need to load from a pretrained checkpoint
        args.resume_from_checkpoint is None 
        and args.pretrained_model_name_or_path is not None
        and os.path.isfile(args.pretrained_model_name_or_path)
    ):
        # Since EMA is deprecated, we do not load EMA from the pretrained checkpoint
        logger.info("Loading from a pretrained checkpoint.")
        checkpoint = torch.load(args.pretrained_model_name_or_path)
        rdp.module.load_state_dict(checkpoint["module"])
   
    # Resume training state from a checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
        # Get the recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path)) # load_module_strict=False
                if accelerator.mixed_precision in ["fp16", "bf16"]:
                    try:
                        # 錯誤的: accelerator.deepspeed_engine
                        # 正確的: rdp
                        resumed_scale = rdp.get_loss_scaler().cur_scale
                        
                        accelerator.print(f"--- [Resumed from Checkpoint] ---")
                        accelerator.print(f"Resumed Dynamic Loss Scale is: {resumed_scale}")
                    except Exception as e:
                        accelerator.print(f"[Warning] Could not read Loss Scaler after resume: {e}")
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = args.learning_rate
                # unwrapped_scheduler = accelerator.unwrap_model(lr_scheduler)
                # if len(optimizer.param_groups) == 3:
                #     new_lrs = [LR_LOW, LR_MID, LR_HIGH]
                #     for i, param_group in enumerate(optimizer.param_groups):
                #         param_group['lr'] = new_lrs[i]
                #     # optimizer.param_groups[0]['lr'] = LR_LOW
                #     # optimizer.param_groups[1]['lr'] = LR_MID
                #     # optimizer.param_groups[2]['lr'] = LR_HIGH
                #     if hasattr(unwrapped_scheduler, "base_lrs"):
                #         unwrapped_scheduler.base_lrs = new_lrs
                #         accelerator.print("Successfully re-applied 3-group LRs to Optimizer and Scheduler base_lrs.")
                #         accelerator.print(f"New Optimizer LRs: {[g['lr'] for g in optimizer.param_groups]}")
                #         accelerator.print(f"New Scheduler Base LRs: {unwrapped_scheduler.base_lrs}")
                #     else:
                #         accelerator.print("[Warning] Scheduler does not have 'base_lrs' attribute. LR re-application might not stick.")
                #     accelerator.print("Successfully loaded state and re-applied 3-group differential LRs.")
                # else:
                #     accelerator.print(f"[Warning] Optimizer has {len(optimizer.param_groups)} param groups. Expected 3. Checkpoint LR restore might be incorrect.")
            except:
                # load deepspeed's state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                rdp.module.load_state_dict(checkpoint["module"])
                # logger.info("Loading state_dict into rdp.module._orig_mod...")
                # rdp.module._orig_mod.load_state_dict(checkpoint["module"])
                
            load_model(ema_rdp, os.path.join(args.output_dir, path, "ema", "model.safetensors"))

            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # show the progress bar only on the main process
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")        

    """
    Training loop
    """
    loss_for_log = {}
    for epoch in range(first_epoch, int(args.num_train_epochs)):

        rdp.train()        
        # Forward and backward...
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(rdp):
                # print(f"==============================================================================================================")
                images = batch["images"].to(dtype=weight_dtype)
                pcds = batch["pointclouds"].to(dtype=torch.float32)
                pcd_mean = batch["pcd_mean"].to(dtype=weight_dtype)
                # pcd_scale_fp32 = batch["pcd_scale"]
                pcd_scale = batch["pcd_scale"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype) # (batch_size, time step, dim)
                keyframes_img = batch["keyframe_images"].to(dtype=weight_dtype)
                keyframes_pcd = batch["keyframe_pcds"].to(dtype=torch.float32)
                keyframes_pcd_mean = batch["keyframe_pcd_mean"].to(dtype=weight_dtype)
                keyframes_pcd_scale = batch["keyframe_pcd_scale"].to(dtype=weight_dtype)
                # keyframe_pcd_scale_fp32 = batch["pcd_scale"]

                
                # We only use the last state as input
                states = states[:, -1:, :]
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                state_elem_mask = state_elem_mask.unsqueeze(1) # (batch_size, 1, dim)
                ctrl_freqs = batch["ctrl_freq"]

                with torch.no_grad():
                    batch_size, img_N , C, H, W = images.shape
                    patch_img_embeds, cls_img_embeds = vision_encoder(images.reshape(-1, C, H, W))
                    patch_img_embeds = patch_img_embeds.reshape(batch_size * img_N, -1, vision_encoder.hidden_size).detach().to(dtype=weight_dtype)
                    cls_img_embeds = cls_img_embeds.reshape(batch_size * img_N, -1, vision_encoder.hidden_size).detach().to(dtype=weight_dtype)

                    _, pcd_N, point_N, _ = pcds.shape
                    global_pcd_embeds, cls_pcd_embeds, patch_pcd_embeds = pcd_encoder(pcds.reshape(-1, point_N, 3))
                    global_pcd_embeds = global_pcd_embeds.reshape(batch_size * pcd_N, -1, pcd_encoder.global_hidden_size).detach().to(dtype=weight_dtype)
                    cls_pcd_embeds = cls_pcd_embeds.reshape(batch_size * pcd_N, -1, pcd_encoder.hidden_size).detach().to(dtype=weight_dtype)
                    patch_pcd_embeds = patch_pcd_embeds.reshape(batch_size * pcd_N, -1, pcd_encoder.hidden_size).detach().to(dtype=weight_dtype)

                    pcd_mean = pcd_mean.reshape(batch_size * pcd_N, -1).to(dtype=weight_dtype)
                    pcd_scale = pcd_scale.reshape(batch_size * pcd_N, -1).to(dtype=weight_dtype)

                    # ==========================================20251027
                    batch_size, keyframe_img_N, keyframe_C, keyframe_H, keyframe_W = keyframes_img.shape
                    # print(f"batch_size {batch_size}")
                    # print(f"keyframe_img_N {keyframe_img_N}")

                    keyframe_patch_img_embeds, keyframe_cls_img_embeds = vision_encoder(keyframes_img.reshape(-1, keyframe_C, keyframe_H, keyframe_W))
                    keyframe_patch_img_embeds = keyframe_patch_img_embeds.reshape(batch_size * keyframe_img_N, -1, vision_encoder.hidden_size).detach().to(dtype=weight_dtype)
                    keyframe_cls_img_embeds = keyframe_cls_img_embeds.reshape(batch_size * keyframe_img_N, -1, vision_encoder.hidden_size).detach().to(dtype=weight_dtype)
                    
                    _, keyframe_pcd_N, keyframe_point_N, _ = keyframes_pcd.shape
                    keyframe_global_pcd_embeds, keyframe_cls_pcd_embeds, keyframe_patch_pcd_embeds = pcd_encoder(keyframes_pcd.reshape(-1, keyframe_point_N, 3))
                    keyframe_global_pcd_embeds = keyframe_global_pcd_embeds.reshape(batch_size * keyframe_pcd_N, -1, pcd_encoder.global_hidden_size).detach().to(dtype=weight_dtype)
                    keyframe_cls_pcd_embeds = keyframe_cls_pcd_embeds.reshape(batch_size * keyframe_pcd_N, -1, pcd_encoder.hidden_size).detach().to(dtype=weight_dtype)
                    keyframe_patch_pcd_embeds = keyframe_patch_pcd_embeds.reshape(batch_size * keyframe_pcd_N, -1, pcd_encoder.hidden_size).detach().to(dtype=weight_dtype)

                    keyframe_pcd_mean = keyframes_pcd_mean.reshape(batch_size * keyframe_pcd_N, -1).to(dtype=weight_dtype)
                    keyframe_pcd_scale = keyframes_pcd_scale.reshape(batch_size * keyframe_pcd_N, -1).to(dtype=weight_dtype)

                del images
                del pcds
                del keyframes_img
                del keyframes_pcd
                    # ==========================================20251027
                
                # print("==================== [Tensor Shape Check] ====================")
                # print("--- [Regular Data] ---")
                # print(f"cls_img_embeds:             {cls_img_embeds.shape}")
                # print(f"patch_img_embeds:           {patch_img_embeds.shape}")
                # print(f"global_pcd_embeds:          {global_pcd_embeds.shape}")
                # print(f"cls_pcd_embeds:             {cls_pcd_embeds.shape}")
                # print(f"patch_pcd_embeds:           {patch_pcd_embeds.shape}")
                # print(f"pcd_mean:                   {pcd_mean.shape}")
                # print(f"pcd_scale:                  {pcd_scale.shape}")
                
                # print("\n--- [Control/GT Data] ---")
                # print(f"state_tokens:               {states.shape}")
                # print(f"action_gt:                  {actions.shape}")
                # print(f"action_mask:                {state_elem_mask.shape}")
                
                # # 這裡假設 ctrl_freqs 可能是個標量或一維張量/列表
                # if isinstance(ctrl_freqs, torch.Tensor):
                #     print(f"ctrl_freqs:                 {ctrl_freqs.shape}")
                # else:
                #     print(f"ctrl_freqs:                 (非 Tensor, 值或列表)")

                # print("\n--- [Keyframe Data] ---")
                # print(f"keyframe_cls_img_embeds:    {keyframe_cls_img_embeds.shape}")
                # print(f"keyframe_patch_img_embeds:  {keyframe_patch_img_embeds.shape}")
                # print(f"keyframe_global_pcd_embeds: {keyframe_global_pcd_embeds.shape}")
                # print(f"keyframe_cls_pcd_embeds:    {keyframe_cls_pcd_embeds.shape}")
                # print(f"keyframe_patch_pcd_embeds:  {keyframe_patch_pcd_embeds.shape}")
                # print(f"keyframe_pcd_mean:          {keyframe_pcd_mean.shape}")
                # print(f"keyframe_pcd_scale:         {keyframe_pcd_scale.shape}")
                # print("==============================================================")
                            
                # train Backbone and fusion
                loss = rdp(
                    cls_img_embeds=cls_img_embeds,
                    patch_img_embeds=patch_img_embeds,
                    global_pcd_embeds=global_pcd_embeds,
                    cls_pcd_embeds=cls_pcd_embeds,
                    patch_pcd_embeds=patch_pcd_embeds,
                    pcd_mean=pcd_mean,
                    pcd_scale=pcd_scale,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_elem_mask,
                    ctrl_freqs=ctrl_freqs,
                    # ==========================================20251027
                    keyframe_cls_img_embeds=keyframe_cls_img_embeds,
                    keyframe_patch_img_embeds=keyframe_patch_img_embeds,
                    keyframe_global_pcd_embeds=keyframe_global_pcd_embeds,
                    keyframe_cls_pcd_embeds=keyframe_cls_pcd_embeds,
                    keyframe_patch_pcd_embeds=keyframe_patch_pcd_embeds,
                    keyframe_pcd_mean=keyframe_pcd_mean,
                    keyframe_pcd_scale=keyframe_pcd_scale,
                    # ==========================================20251027
                )
                if torch.isnan(loss):
                    accelerator.print("[CRITICAL ERROR] Loss is NaN. Aborting this step for stability.")
                    continue

                if torch.isinf(loss):
                    accelerator.print("[CRITICAL ERROR] Loss is INF. Aborting this step for stability.")
                    continue
                # calculate the gradients and update the parameters
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = rdp.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                # update the parameters
                optimizer.step()
                # update the learning rate
                lr_scheduler.step()
                # reset the gradients
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            
            # use the exponential moving average to update the parameters
            ema_model.step(accelerator.unwrap_model(rdp))

            if accelerator.sync_gradients:
                # update the progress bar and train step
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.mixed_precision in ["fp16", "bf16"]:
                        try:
                            # 正確的: rdp
                            current_scale = rdp.get_loss_scaler().cur_scale
                            accelerator.print(f"--- [Saving Checkpoint] ---")
                            accelerator.print(f"Current Dynamic Loss Scale is: {current_scale}")
                        except Exception as e:
                            accelerator.print(f"[Warning] Could not read Loss Scaler before save: {e}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, f"ema")
                    accelerator.save_model(ema_rdp, ema_save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        vision_encoder,
                        pcd_encoder,
                        rdp,
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataloader,
                        logger,
                        )
                    # recode the loss for logging
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)
            # show the loss and learning rate on the progress bar
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr']}
            # logs = {
            #     "loss": loss.detach().item(), 
            #     "lr_low": optimizer.param_groups[0]['lr'],
            #     "lr_mid": optimizer.param_groups[1]['lr'],
            #     "lr_high": optimizer.param_groups[2]['lr']
            # }
            logs = {
                "loss": loss.detach().item(), 
                "lr_low": lr_scheduler.get_last_lr()[0],
                # "lr_mid": lr_scheduler.get_last_lr()[1],
                "lr_high": lr_scheduler.get_last_lr()[1]
            }


            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            # log the loss and learning rate
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

    # check all processes have finished
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save the model to the output directory
        accelerator.unwrap_model(rdp).save_pretrained(args.output_dir)
        # save the EMA model to the output directory
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_rdp, ema_save_path)
        
        logger.info(f"Saved Model to {args.output_dir}")
    accelerator.end_training()
