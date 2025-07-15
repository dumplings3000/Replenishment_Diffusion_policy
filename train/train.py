#!/usr/bin/env python
# coding=utf-8
import copy
import logging
import math
import os
from pathlib import Path

import diffusers
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm
from safetensors.torch import load_model

from models.ema_model import EMAModel
from models.multimodal_encoder.VIT_encoder import VITEncoder
from models.multimodal_encoder.PCD_encoder import PCDEncoder
from models.rdp_runner import RDPRunner
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
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None,
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

    vision_encoder = VITEncoder(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor
    pcd_encoder = PCDEncoder(pcd_tower=args.pretrained_pcd_encoder_name_or_path, args=None)
    pcd_processor = pcd_encoder.pcd_processor

    # Load from a pretrained checkpoint
    if (
        args.pretrained_model_name_or_path is not None
        and not os.path.isfile(args.pretrained_model_name_or_path)
    ):
        logger.info("Constructing model from pretrained checkpoint.")
        rdp = RDPRunner.from_pretrained(args.pretrained_model_name_or_path)
    else:
        logger.info("Constructing model from provided config.")
        # Calculate the image condition length
        img_cond_len = (config["common"]["img_history_size"] 
                        * config["common"]["num_cameras"] 
                        * vision_encoder.num_patches)
        pcd_cond_len = (config["common"]["pcd_history_size"] 
                        * config["common"]["num_cameras"]
                        * pcd_encoder.num_patches)
        rdp = RDPRunner(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            img_token_dim=config["model"]["img_token_dim"],
            pcd_token_dim=config["model"]["pcd_token_dim"],
            state_token_dim=config["model"]["state_token_dim"],
            img_cond_len=img_cond_len,
            pcd_cond_len=pcd_cond_len,
            img_pos_embed_config=[
                ("image", (config["common"]["img_history_size"], 
                    config["common"]["num_cameras"], 
                    -vision_encoder.num_patches)),  
            ],
            pcd_pos_embed_config=[
                ("pcd", (config["common"]["pcd_history_size"], 
                    config["common"]["num_cameras"], 
                    -pcd_encoder.num_patches)),  
            ],
            dtype=weight_dtype,
        )
        
    # EMA model, Use to update the model weights with exponential moving average, It's can help to improve the performance of the model.                                      
    ema_rdp = copy.deepcopy(rdp)
    ema_model = EMAModel(ema_rdp, 
                        update_after_step=config["model"]["ema"]["update_after_step"],
                        inv_gamma=config["model"]["ema"]["inv_gamma"],
                        power=config["model"]["ema"]["power"],
                        min_val=config["model"]["ema"]["min_val"],
                        max_val=config["model"]["ema"]["max_val"])
    
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

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdp))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)
    
    # if args.gradient_checkpointing:
    #     # TODO: 
    #     raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # if args.scale_lr:
    #     args.learning_rate = (
    #         args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    #     )


    # Optimizer creation
    params_to_optimize = rdp.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    """
    ===========================================DATA Loader, batching, and collating====================================
    """
    
    # Dataset and DataLoaders creation:                                                           
    train_dataset = VAConsumerDataset(
        config=config["dataset"],
        image_processor=image_processor,
        pcd_processor=pcd_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
    )
    sample_dataset = VAConsumerDataset(
        config=config["dataset"],
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
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
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with accelerator.
    rdp, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        rdp, optimizer, train_dataloader, sample_dataloader, lr_scheduler                   
    )
    ema_rdp.to(accelerator.device, dtype=weight_dtype)                                                          
    
    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)
    
    if pcd_encoder is not None:
        pcd_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

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
            except:
                # load deepspeed's state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                rdp.module.load_state_dict(checkpoint["module"])
                
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
        
        # Set the progress_bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)
        
        # Forward and backward...
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(rdp):
                images = batch["images"].to(dtype=weight_dtype)
                pcds = batch["pcds"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype) # (batch_size, time step, dim)
                # We only use the last state as input
                states = states[:, -1:, :]
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                state_elem_mask = state_elem_mask.unsqueeze(1) # (batch_size, 1, dim)
                ctrl_freqs = batch["ctrl_freqs"]
                
                # Vision Encoder forward pass
                with torch.no_grad():
                    batch_size, _ , C, H, W = images.shape
                    img_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    img_embeds = img_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))
                    pcd_embeds = pcd_encoder(pcds.reshape(-1, 3)).detach()
                    pcd_embeds = pcd_embeds.reshape((batch_size, -1, pcd_encoder.hidden_size))
                
                # train Backbone and adaptor
                loss = rdp(
                    img_tokens=img_embeds,
                    pcd_tokens=pcd_embeds,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_elem_mask,
                    ctrl_freqs=ctrl_freqs
                )
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
                optimizer.zero_grad(set_to_none=args.set_grad_to_none)
            
            # use the exponential moving average to update the parameters
            ema_model.step(accelerator.unwrap_model(rdp))

            if accelerator.sync_gradients:
                # update the progress bar and train step
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
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
                        sample_dataset.get_dataset_id2name(),
                        sample_dataloader,
                        logger,
                        )
                    # recode the loss for logging
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)
            # show the loss and learning rate on the progress bar
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
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
