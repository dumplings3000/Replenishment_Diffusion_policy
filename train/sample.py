from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def log_sample_res(
    vision_encoder, pcd_encoder, rdp, args, 
    accelerator, weight_dtype, dataloader, logger
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )

    rdp.eval()
    
    loss_for_log = defaultdict(float)
    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break
        
        ctrl_freqs = batch["ctrl_freq"].to(dtype=weight_dtype)
        state_norm = batch["state_norm"].to(dtype=weight_dtype)
        images = batch["images"].to(dtype=weight_dtype)
        pcds = batch["pointclouds"].to(dtype=weight_dtype)
        pcd_mean = batch["pcd_mean"].to(dtype=weight_dtype)
        pcd_scale = batch["pcd_scale"].to(dtype=weight_dtype)
        states = batch["states"].to(dtype=weight_dtype)

        keyframes_img = batch["keyframe_images"].to(dtype=weight_dtype)
        keyframes_pcd = batch["keyframe_pcds"].to(dtype=torch.float32) # <-- 保持 FP32
        keyframes_pcd_mean = batch["keyframe_pcd_mean"].to(dtype=weight_dtype)
        keyframes_pcd_scale = batch["keyframe_pcd_scale"].to(dtype=weight_dtype)
        # We only use the last state as input
        states = states[:, -1:, :]
        actions = batch["actions"].to(dtype=weight_dtype)
        state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
            
        batch_size, img_N , C, H, W = images.shape
        patch_img_embeds, cls_img_embeds = vision_encoder(images.reshape(-1, C, H, W))
        patch_img_embeds = patch_img_embeds.reshape(batch_size * img_N, -1, vision_encoder.hidden_size).detach().to(dtype=weight_dtype)
        cls_img_embeds = cls_img_embeds.reshape(batch_size * img_N, -1, vision_encoder.hidden_size).detach().to(dtype=weight_dtype)
        _, pcd_N, point_N, _ = pcds.shape
        global_pcd_embeds, cls_pcd_embeds, patch_pcd_embeds = pcd_encoder(pcds.reshape(-1, point_N, 3).float())
        global_pcd_embeds = global_pcd_embeds.reshape(batch_size * pcd_N, -1, pcd_encoder.global_hidden_size).detach().to(dtype=weight_dtype)
        cls_pcd_embeds = cls_pcd_embeds.reshape(batch_size * pcd_N, -1, pcd_encoder.hidden_size).detach().to(dtype=weight_dtype)
        patch_pcd_embeds = patch_pcd_embeds.reshape(batch_size * pcd_N, -1, pcd_encoder.hidden_size).detach().to(dtype=weight_dtype)
        pcd_mean = pcd_mean.reshape(batch_size * pcd_N, -1).to(dtype=weight_dtype)
        pcd_scale = pcd_scale.reshape(batch_size * pcd_N, -1).to(dtype=weight_dtype)

        # --- 新增 Keyframe 編碼 ---
        batch_size, keyframe_img_N, keyframe_C, keyframe_H, keyframe_W = keyframes_img.shape

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
            
        # pred_actions = rdp.predict_action(
        #     cls_img_embeds=cls_img_embeds,
        #     patch_img_embeds=patch_img_embeds,
        #     global_pcd_embeds=global_pcd_embeds,
        #     cls_pcd_embeds=cls_pcd_embeds,
        #     patch_pcd_embeds=patch_pcd_embeds,
        #     pcd_mean=pcd_mean,
        #     pcd_scale=pcd_scale,
        #     state_tokens=states,
        #     action_mask=state_elem_mask.unsqueeze(1),
        #     ctrl_freqs=ctrl_freqs
        # )
        pred_actions = rdp.predict_action(
            cls_img_embeds=cls_img_embeds,
            patch_img_embeds=patch_img_embeds,
            global_pcd_embeds=global_pcd_embeds,
            cls_pcd_embeds=cls_pcd_embeds,
            patch_pcd_embeds=patch_pcd_embeds,
            pcd_mean=pcd_mean,
            pcd_scale=pcd_scale,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
            # --- 傳入 Keyframe 參數 ---
            keyframe_cls_img_embeds=keyframe_cls_img_embeds,
            keyframe_patch_img_embeds=keyframe_patch_img_embeds,
            keyframe_global_pcd_embeds=keyframe_global_pcd_embeds,
            keyframe_cls_pcd_embeds=keyframe_cls_pcd_embeds,
            keyframe_patch_pcd_embeds=keyframe_patch_pcd_embeds,
            keyframe_pcd_mean=keyframe_pcd_mean,
            keyframe_pcd_scale=keyframe_pcd_scale,
        )
        num_steps = pred_actions.shape[1]
        expanded_state_elem_mask = state_elem_mask.unsqueeze(1).tile((1, num_steps, 1)).float()
        expanded_state_norm = state_norm.unsqueeze(1).tile((1, num_steps, 1)).float()
        
        loss = F.mse_loss(pred_actions, actions, reduction='none').float()
        
        mse_loss = (loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
        mse_loss_scaler = accelerator.gather(mse_loss).mean().item()
        loss_for_log["overall_avg_sample_mse"] += mse_loss_scaler
        
        l2_loss = loss.sqrt() / (expanded_state_norm + 1e-3)
        l2_loss = (l2_loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
        l2_loss_scaler = accelerator.gather(l2_loss).mean().item()
        loss_for_log["overall_avg_sample_l2err"] += l2_loss_scaler

    # 平均化
    loss_for_log["overall_avg_sample_mse"] = round(loss_for_log["overall_avg_sample_mse"] / args.num_sample_batches, 4)
    loss_for_log["overall_avg_sample_l2err"] = round(loss_for_log["overall_avg_sample_l2err"] / args.num_sample_batches, 4)
    print("pred_actions min/max:", pred_actions.min().item(), pred_actions.max().item())
    print("actions min/max:", actions.min().item(), actions.max().item())
    print("expanded_state_elem_mask sum:", expanded_state_elem_mask.sum().item())
    print("expanded_state_norm min/max:", expanded_state_norm.min().item(), expanded_state_norm.max().item())

    rdp.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)