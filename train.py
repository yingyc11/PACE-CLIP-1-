import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from dcpg import DCPGPromptGenerator
from pabef import PABEFFusion
from dgdl import shallow_deep_semantic_enhancement
from loss import FocalLoss, BinaryDiceLoss
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from sklearn.metrics import average_precision_score


# ============================================================
# PACE-CLIP training implementation

#
# Base: Hybrid semantic router.
# PACE-CLIP training branches:
#   1) Pixel branch has its own PromptLearner + shallow cross-attn.
#      It is optimized only by pixel-level Focal/Dice loss.
#   2) Image branch has its own PromptLearner + shallow cross-attn.
#      It is optimized only by image-level BCE loss + evidence balance.
#   3) CLIP backbone is frozen. DynamicFusionMLP is removed.
#   4) Test uses pixel-branch map for pixel metrics and image-branch
#      balanced evidence score for image metrics.
# ============================================================




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_ap_score(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def safe_nanmean(values):
    values = [v for v in values if not np.isnan(v)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def linear_warmup(epoch_idx, total_epoch, start_value, end_value):
    """epoch_idx is 0-based."""
    if total_epoch <= 1:
        return float(end_value)
    progress = float(epoch_idx) / float(max(1, total_epoch - 1))
    return float(start_value + progress * (end_value - start_value))


def delayed_warmup(epoch_idx, total_epoch, delay_epochs, start_value, end_value):
    """
    Image branch delay. During first delay_epochs, lambda_image=0.
    Then linearly warm up from start_value to end_value.
    """
    if epoch_idx < delay_epochs:
        return 0.0
    remain = max(1, total_epoch - delay_epochs)
    local_idx = epoch_idx - delay_epochs
    return linear_warmup(local_idx, remain, start_value, end_value)


def anomaly_score_from_similarity_map(similarity_map):
    """
    similarity_map: [B, 2, H, W]
    return anomaly map: [B, H, W]
    """
    anomaly_map = (similarity_map[:, 1, :, :] + 1.0 - similarity_map[:, 0, :, :]) / 2.0
    return torch.clamp(anomaly_map, min=1e-6, max=1.0 - 1e-6)


def topk_score_from_map(anomaly_map, topk_ratio=0.0001):
    if anomaly_map.dim() == 4:
        anomaly_map = anomaly_map.squeeze(1)
    b = anomaly_map.shape[0]
    flat = anomaly_map.reshape(b, -1)
    k = max(1, int(flat.shape[1] * float(topk_ratio)))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)


def compute_global_anomaly_score(image_features, text_features, tau=0.07):
    """
    image_features: [B, D], normalized
    text_features: [1+K, D], normalized; index 0 normal, index 1: abnormal prompts
    return global anomaly probability [B]
    """
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    normal_feature = text_features[0:1]
    anomaly_candidates = text_features[1:]

    sim_n = torch.matmul(image_features, normal_feature.t())  # [B,1]
    sim_a_all = torch.matmul(image_features, anomaly_candidates.t())  # [B,K]
    semantic_w = F.softmax(sim_a_all / tau, dim=-1)
    sim_a = (semantic_w * sim_a_all).sum(dim=-1, keepdim=True)
    logits = torch.cat([sim_n, sim_a], dim=1) / tau
    return torch.clamp(F.softmax(logits, dim=1)[:, 1], min=1e-6, max=1.0 - 1e-6)


def compute_hybrid_semantic_map(
    patch_feature,
    text_features,
    image_features,
    image_size,
    lambda_router=0.7,
    patch_ratio=0.25,
    image_router_tau=0.07,
    patch_router_tau=0.1,
    map_tau=0.07,
):
    """
    Hybrid semantic router.

    final abnormal similarity =
        (1 - patch_ratio) * image-wise abnormal similarity
        + patch_ratio * patch-wise abnormal similarity

    patch_feature: [B, N, D]
    text_features: [1+K, D]
    image_features: [B, D]

    return: [B, 2, H, W]
    """
    B = patch_feature.size(0)

    patch_feature = F.normalize(patch_feature, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)

    normal_feature = text_features[0:1]
    anomaly_candidates = text_features[1:]

    image_logits = torch.matmul(image_features, anomaly_candidates.t()) / image_router_tau
    image_weights = F.softmax(image_logits, dim=-1)
    abnormal_image = torch.matmul(image_weights, anomaly_candidates)

    abnormal_mean = anomaly_candidates.mean(dim=0, keepdim=True).expand(B, -1)
    abnormal_image = lambda_router * abnormal_image + (1.0 - lambda_router) * abnormal_mean
    abnormal_image = F.normalize(abnormal_image, dim=-1)

    sim_map_list = []
    for b in range(B):
        pf = patch_feature[b:b + 1]
        sim_n = torch.matmul(pf, normal_feature.t())
        sim_a_image = torch.matmul(pf, abnormal_image[b:b + 1].t())
        sim_a_all = torch.matmul(pf, anomaly_candidates.t())
        semantic_w = F.softmax(sim_a_all / patch_router_tau, dim=-1)
        sim_a_patch = (semantic_w * sim_a_all).sum(dim=-1, keepdim=True)
        sim_a = (1.0 - patch_ratio) * sim_a_image + patch_ratio * sim_a_patch
        sim = torch.cat([sim_n, sim_a], dim=-1)
        sim = F.softmax(sim / map_tau, dim=-1)
        similarity_map = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], image_size).permute(0, 3, 1, 2)
        sim_map_list.append(similarity_map)
    return torch.cat(sim_map_list, dim=0)



def compute_branch_maps(
    patch_features,
    image_features,
    text_features,
    cross_attn,
    args,
    patch_ratio,
    lambda_router,
):
    patch_features_optimized = shallow_deep_semantic_enhancement(patch_features, cross_attn, args.gamma_ca)
    maps = []
    for idx, patch_feature in enumerate(patch_features_optimized):
        if idx in args.feature_map_layer:
            sim_map = compute_hybrid_semantic_map(
                patch_feature=patch_feature,
                text_features=text_features,
                image_features=image_features,
                image_size=args.image_size,
                lambda_router=lambda_router,
                patch_ratio=patch_ratio,
                image_router_tau=0.07,
                patch_router_tau=args.patch_router_tau,
                map_tau=0.07,
            )
            maps.append(sim_map)
    if len(maps) != 4:
        raise RuntimeError(
            f"PACE-CLIP expects 4 similarity maps, got {len(maps)}. "
            f"Check --features_list and --feature_map_layer."
        )
    return maps


def load_resume(args, modules, optimizer, device):
    start_epoch = 0
    if args.resume_path is None:
        return start_epoch

    print(f"Resume PACE-CLIP training from: {args.resume_path}")
    ckpt = torch.load(args.resume_path, map_location=device)

    key_map = {
        "prompt_learner_pixel": "prompt_learner_pixel",
        "cross_attn_pixel": "cross_attn_pixel",
        "prompt_learner_image": "prompt_learner_image",
        "cross_attn_image": "cross_attn_image",
        "evidence_fusion": "evidence_fusion",
    }
    for module_name, key in key_map.items():
        if key in ckpt:
            modules[module_name].load_state_dict(ckpt[key], strict=False)
        else:
            print(f"Warning: missing {key} in checkpoint; keep current initialization.")

    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("Optimizer state loaded.")
        except Exception as exc:
            print(f"Warning: failed to load optimizer state: {exc}")

    if "epoch" in ckpt:
        start_epoch = int(ckpt["epoch"])
    else:
        import re
        match = re.search(r"epoch_(\d+)", os.path.basename(args.resume_path))
        start_epoch = int(match.group(1)) if match else 0
    print(f"Start from epoch {start_epoch + 1}")
    return start_epoch


def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_path, exist_ok=True)

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
        "cpg_topk_ratio": args.cpg_topk_ratio,
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform,
                         dataset_name=args.dataset, mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Two independent granularity-specific branches.
    prompt_learner_pixel = DCPGPromptGenerator(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner_image = DCPGPromptGenerator(model.to("cpu"), AnomalyCLIP_parameters)

    visual_dim = 768
    cross_attn_pixel = torch.nn.MultiheadAttention(embed_dim=visual_dim, num_heads=4, batch_first=True).to(device)
    cross_attn_image = torch.nn.MultiheadAttention(embed_dim=visual_dim, num_heads=4, batch_first=True).to(device)

    evidence_fusion = PABEFFusion(
        num_evidence=5,
        temperature=args.fusion_temperature,
        floor_ratio=args.fusion_floor_ratio,
    ).to(device)

    prompt_learner_pixel.to(device)
    prompt_learner_image.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    # Freeze CLIP backbone. Train only lightweight prompt/cross/fusion branches.
    for p in model.parameters():
        p.requires_grad = False
    for module in [prompt_learner_pixel, prompt_learner_image, cross_attn_pixel, cross_attn_image, evidence_fusion]:
        for p in module.parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": prompt_learner_pixel.parameters(), "lr": args.pixel_prompt_lr, "weight_decay": args.weight_decay},
            {"params": cross_attn_pixel.parameters(), "lr": args.pixel_cross_lr, "weight_decay": args.weight_decay},
            {"params": prompt_learner_image.parameters(), "lr": args.image_prompt_lr, "weight_decay": args.weight_decay},
            # Kept for checkpoint/resume compatibility; the local image maps do not use it.
            {"params": cross_attn_image.parameters(), "lr": args.image_cross_lr, "weight_decay": args.weight_decay},
            {"params": evidence_fusion.parameters(), "lr": args.fusion_lr, "weight_decay": 0.0},
        ],
        betas=(0.5, 0.999),
    )

    modules = {
        "prompt_learner_pixel": prompt_learner_pixel,
        "cross_attn_pixel": cross_attn_pixel,
        "prompt_learner_image": prompt_learner_image,
        "cross_attn_image": cross_attn_image,
        "evidence_fusion": evidence_fusion,
    }
    start_epoch = load_resume(args, modules, optimizer, device)

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    for epoch in tqdm(range(start_epoch, args.epoch)):
        model.eval()
        prompt_learner_pixel.train()
        cross_attn_pixel.train()
        prompt_learner_image.train()
        cross_attn_image.train()
        evidence_fusion.train()

        lambda_image_now = delayed_warmup(
            epoch, args.epoch, args.image_delay_epochs,
            args.lambda_image_start, args.lambda_image_end,
        )

        loss_list, pixel_loss_list, image_loss_list, balance_loss_list = [], [], [], []
        train_pap_list, train_iap_list = [], []
        pos_score_list, neg_score_list, score_gap_list = [], [], []
        image_w_list = []

        for items in tqdm(train_dataloader):
            image = items["img"].to(device)
            label = items["anomaly"].float().to(device)
            gt = items["img_mask"].squeeze().to(device)
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if len(patch_features) < 4:
                raise RuntimeError(
                    f"Expected at least 4 patch feature maps, got {len(patch_features)}. "
                    f"Please use --features_list 6 12 18 24. Current args.features_list={args.features_list}"
                )

            # ---------------- Pixel branch ----------------
            deep_patch_features_pixel = patch_features[3].detach()
            prompts_p, tokenized_p, compound_p = prompt_learner_pixel(deep_patch_features_pixel, cls_id=None)
            text_features_pixel = model.encode_text_learn(prompts_p, tokenized_p, compound_p).float()
            text_features_pixel = F.normalize(text_features_pixel, dim=-1)

            pixel_maps = compute_branch_maps(
                patch_features=patch_features,
                image_features=image_features,
                text_features=text_features_pixel,
                cross_attn=cross_attn_pixel,
                args=args,
                patch_ratio=args.pixel_patch_ratio,
                lambda_router=args.lambda_router,
            )

            pixel_loss = 0.0
            for sim_map in pixel_maps:
                pixel_loss = pixel_loss + loss_focal(sim_map, gt)
                pixel_loss = pixel_loss + loss_dice(sim_map[:, 1, :, :], gt)
                pixel_loss = pixel_loss + loss_dice(sim_map[:, 0, :, :], 1 - gt)

            fused_pixel_map = torch.stack(pixel_maps, dim=0).mean(dim=0)
            pixel_loss = pixel_loss + loss_focal(fused_pixel_map, gt)
            pixel_loss = pixel_loss + loss_dice(fused_pixel_map[:, 1, :, :], gt)
            pixel_loss = pixel_loss + loss_dice(fused_pixel_map[:, 0, :, :], 1 - gt)
            anomaly_map_pixel = anomaly_score_from_similarity_map(fused_pixel_map)

            # ---------------- Image branch ----------------
            deep_patch_features_image = patch_features[3].detach()
            prompts_i, tokenized_i, compound_i = prompt_learner_image(deep_patch_features_image, cls_id=None)
            text_features_image = model.encode_text_learn(prompts_i, tokenized_i, compound_i).float()
            text_features_image = F.normalize(text_features_image, dim=-1)

            # PACE-CLIP implementation:
            # Image branch only provides global recognition.
            # Local image evidence is anchored by pixel branch maps and detached,
            # so image loss cannot corrupt pixel localization.
            global_score = compute_global_anomaly_score(image_features, text_features_image)
            layer_topk_scores = [
                topk_score_from_map(
                    anomaly_score_from_similarity_map(m.detach()),
                    topk_ratio=args.topk_ratio,
                )
                for m in pixel_maps
            ]
            image_scores_train, image_w_bal = evidence_fusion(global_score, layer_topk_scores)
            image_loss = F.binary_cross_entropy(image_scores_train, label)
            loss_balance = evidence_fusion.balance_loss()

            total_loss = (
                args.lambda_pixel * pixel_loss
                + lambda_image_now * image_loss
                + args.lambda_balance * loss_balance
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append(total_loss.detach().item())
            pixel_loss_list.append(pixel_loss.detach().item())
            image_loss_list.append(image_loss.detach().item())
            balance_loss_list.append(loss_balance.detach().item())
            image_w_list.append(image_w_bal.detach().cpu().numpy())

            with torch.no_grad():
                label_np = label.detach().cpu().numpy()
                img_score_np = image_scores_train.detach().cpu().numpy()
                train_iap_list.append(safe_ap_score(label_np, img_score_np))

                pix_gt_np = gt.detach().cpu().numpy().reshape(-1).astype(np.uint8)
                pix_score_np = anomaly_map_pixel.detach().cpu().numpy().reshape(-1)
                train_pap_list.append(safe_ap_score(pix_gt_np, pix_score_np))

                pos_mask = label > 0.5
                neg_mask = label <= 0.5
                if pos_mask.any():
                    pos_score_list.append(image_scores_train[pos_mask].mean().detach().item())
                if neg_mask.any():
                    neg_score_list.append(image_scores_train[neg_mask].mean().detach().item())
                if pos_mask.any() and neg_mask.any():
                    gap = image_scores_train[pos_mask].mean() - image_scores_train[neg_mask].mean()
                    score_gap_list.append(gap.detach().item())

        if (epoch + 1) % args.print_freq == 0:
            image_w_mean = np.mean(np.stack(image_w_list, axis=0), axis=0) if len(image_w_list) else np.zeros(5)
            image_w_str = [round(float(x), 3) for x in image_w_mean.tolist()]
            logger.info(
                "epoch [{}/{}], total_loss:{:.4f}, pixel_loss:{:.4f}, image_loss:{:.4f}, "
                "lambda_image:{:.3f}, balance_loss:{:.4f}, train_pap:{:.2f}, train_iap:{:.2f}, "
                "pos_score:{:.4f}, neg_score:{:.4f}, gap:{:.4f}, image_w_bal:{}".format(
                    epoch + 1, args.epoch, np.mean(loss_list), np.mean(pixel_loss_list),
                    np.mean(image_loss_list), lambda_image_now, np.mean(balance_loss_list),
                    safe_nanmean(train_pap_list) * 100, safe_nanmean(train_iap_list) * 100,
                    safe_nanmean(pos_score_list), safe_nanmean(neg_score_list),
                    safe_nanmean(score_gap_list), image_w_str,
                )
            )

        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_path, f"epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "prompt_learner_pixel": prompt_learner_pixel.state_dict(),
                "cross_attn_pixel": cross_attn_pixel.state_dict(),
                "prompt_learner_image": prompt_learner_image.state_dict(),
                "cross_attn_image": cross_attn_image.state_dict(),
                "evidence_fusion": evidence_fusion.state_dict(),
                "gamma_ca": args.gamma_ca,
                "lambda_router": args.lambda_router,
                "topk_ratio": args.topk_ratio,
                "cpg_topk_ratio": args.cpg_topk_ratio,
                "patch_router_tau": args.patch_router_tau,
                "image_local_evidence": "pixel_maps_detach",
                "pixel_patch_ratio": args.pixel_patch_ratio,
                "image_patch_ratio": args.image_patch_ratio,
                "lambda_pixel": args.lambda_pixel,
                "lambda_image_start": args.lambda_image_start,
                "lambda_image_end": args.lambda_image_end,
                "lambda_balance": args.lambda_balance,
                "fusion_floor_ratio": args.fusion_floor_ratio,
                "fusion_temperature": args.fusion_temperature,
                "image_delay_epochs": args.image_delay_epochs,
                "depth": args.depth,
                "n_ctx": args.n_ctx,
                "t_n_ctx": args.t_n_ctx,
                "features_list": args.features_list,
                "feature_map_layer": args.feature_map_layer,
            }, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PACE-CLIP training", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa")
    parser.add_argument("--save_path", type=str, default="./checkpoint")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--resume_path", type=str, default=None)

    parser.add_argument("--pixel_prompt_lr", type=float, default=5e-5)
    parser.add_argument("--pixel_cross_lr", type=float, default=5e-5)
    parser.add_argument("--image_prompt_lr", type=float, default=5e-5)
    parser.add_argument("--image_cross_lr", type=float, default=5e-5)
    parser.add_argument("--fusion_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--gamma_ca", type=float, default=0.1)
    parser.add_argument("--lambda_router", type=float, default=0.7)
    parser.add_argument("--topk_ratio", type=float, default=0.005)
    parser.add_argument("--cpg_topk_ratio", type=float, default=0.05)
    parser.add_argument("--patch_router_tau", type=float, default=0.1)
    parser.add_argument("--pixel_patch_ratio", type=float, default=0.25)
    parser.add_argument("--image_patch_ratio", type=float, default=0.25)

    parser.add_argument("--lambda_pixel", type=float, default=4.0)
    parser.add_argument("--lambda_image_start", type=float, default=0.3)
    parser.add_argument("--lambda_image_end", type=float, default=0.5)
    parser.add_argument("--lambda_balance", type=float, default=0.03)
    parser.add_argument("--fusion_floor_ratio", type=float, default=0.10)
    parser.add_argument("--fusion_temperature", type=float, default=1.0)
    parser.add_argument("--image_delay_epochs", type=int, default=0)

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
