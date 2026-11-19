import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from dcpg import DCPGPromptGenerator
from pabef import PABEFFusion
from dgdl import shallow_deep_semantic_enhancement
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score
import os




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pixel_average_precision(gt_mask, anomaly_map):
    gt = gt_mask
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()
    pred = anomaly_map
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    gt = gt.astype(np.uint8).reshape(-1)
    pred = pred.reshape(-1)
    if gt.sum() == 0:
        return np.nan
    return average_precision_score(gt, pred)


def anomaly_score_from_similarity_map(similarity_map):
    """
    train format: [B, 2, H, W]
    test format:  [B, H, W, 2]
    return: [B, H, W]
    """
    if similarity_map.dim() != 4:
        raise ValueError(f"Expected 4-D similarity_map, got {similarity_map.shape}")
    if similarity_map.shape[1] == 2:
        anomaly_map = (similarity_map[:, 1, :, :] + 1.0 - similarity_map[:, 0, :, :]) / 2.0
    elif similarity_map.shape[-1] == 2:
        anomaly_map = (similarity_map[..., 1] + 1.0 - similarity_map[..., 0]) / 2.0
    else:
        raise ValueError(f"Cannot infer channel dimension for shape {similarity_map.shape}")
    return torch.clamp(anomaly_map, min=1e-6, max=1.0 - 1e-6)


def topk_score_from_map(anomaly_map, topk_ratio=0.0001):
    if anomaly_map.dim() == 4:
        anomaly_map = anomaly_map.squeeze(1)
    b = anomaly_map.shape[0]
    flat = anomaly_map.reshape(b, -1)
    k = max(1, int(flat.shape[1] * float(topk_ratio)))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)


def compute_global_anomaly_score(image_features, text_features, tau=0.07):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    normal_feature = text_features[0:1]
    anomaly_candidates = text_features[1:]

    sim_n = torch.matmul(image_features, normal_feature.t())
    sim_a_all = torch.matmul(image_features, anomaly_candidates.t())
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
    """Hybrid semantic router for test. return: [B,H,W,2]"""
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
        similarity_map = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], image_size)
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
            f"PACE-CLIP expects 4 similarity maps, got {len(maps)}. Check --features_list and --feature_map_layer."
        )
    return maps


def test(args):
    logger = get_logger(args.save_path)
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

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                        dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    for obj in obj_list:
        results[obj] = {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []}

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

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    required_keys = [
        "prompt_learner_pixel", "cross_attn_pixel",
        "prompt_learner_image", "cross_attn_image", "evidence_fusion",
    ]
    for k in required_keys:
        if k not in checkpoint:
            raise KeyError(f"Missing '{k}' in checkpoint. Please test an PACE-CLIP checkpoint.")

    prompt_learner_pixel.load_state_dict(checkpoint["prompt_learner_pixel"], strict=False)
    cross_attn_pixel.load_state_dict(checkpoint["cross_attn_pixel"], strict=False)
    prompt_learner_image.load_state_dict(checkpoint["prompt_learner_image"], strict=False)
    cross_attn_image.load_state_dict(checkpoint["cross_attn_image"], strict=False)
    evidence_fusion.load_state_dict(checkpoint["evidence_fusion"], strict=False)

    prompt_learner_pixel.to(device)
    prompt_learner_image.to(device)
    cross_attn_pixel.to(device)
    cross_attn_image.to(device)
    evidence_fusion.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    for module in [prompt_learner_pixel, prompt_learner_image, cross_attn_pixel, cross_attn_image, evidence_fusion]:
        module.eval()

    image_w_collect = []

    for _, items in enumerate(tqdm(test_dataloader)):
        image = items["img"].to(device)
        cls_name = items["cls_name"]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]["imgs_masks"].append(gt_mask)
        results[cls_name[0]]["gt_sp"].extend(items["anomaly"].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if len(patch_features) < 4:
                raise RuntimeError(
                    f"Expected at least 4 patch feature maps, got {len(patch_features)}. "
                    f"Please use --features_list 6 12 18 24. Current args.features_list={args.features_list}"
                )

            # Pixel branch for pixel localization.
            prompts_p, tokenized_p, compound_p = prompt_learner_pixel(patch_features[3], cls_id=None)
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
            fused_pixel_map = torch.stack(pixel_maps, dim=0).mean(dim=0)  # [B,H,W,2]
            anomaly_map = anomaly_score_from_similarity_map(fused_pixel_map)

            # Image branch for image classification.
            prompts_i, tokenized_i, compound_i = prompt_learner_image(patch_features[3], cls_id=None)
            text_features_image = model.encode_text_learn(prompts_i, tokenized_i, compound_i).float()
            text_features_image = F.normalize(text_features_image, dim=-1)
            # PACE-CLIP implementation:
            # Image branch only computes global recognition.
            # Local image evidence comes from detached pixel branch maps.
            global_score = compute_global_anomaly_score(image_features, text_features_image)
            layer_topk_scores = [
                topk_score_from_map(
                    anomaly_score_from_similarity_map(m.detach()),
                    topk_ratio=args.topk_ratio,
                )
                for m in pixel_maps
            ]
            image_score, image_w = evidence_fusion(global_score, layer_topk_scores)
            image_w_collect.append(image_w.detach().cpu().numpy())

            results[cls_name[0]]["pr_sp"].extend(image_score.detach().cpu())
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0
            )
            results[cls_name[0]]["anomaly_maps"].append(anomaly_map)

    if len(image_w_collect) > 0:
        image_w_mean = np.mean(np.stack(image_w_collect, axis=0), axis=0)
        logger.info("PACE-CLIP image_w_bal [global, pixel_L1, pixel_L2, pixel_L3, pixel_L4]: %s",
                    [round(float(x), 4) for x in image_w_mean.tolist()])

    table_ls = []
    image_auroc_list, image_ap_list = [], []
    pixel_auroc_list, pixel_aupro_list, pixel_ap_list = [], [], []

    for obj in obj_list:
        table = [obj]
        results[obj]["imgs_masks"] = torch.cat(results[obj]["imgs_masks"])
        results[obj]["anomaly_maps"] = torch.cat(results[obj]["anomaly_maps"]).detach().cpu().numpy()
        pixel_ap = pixel_average_precision(results[obj]["imgs_masks"], results[obj]["anomaly_maps"])

        if args.metrics == "image-level":
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table += [str(np.round(image_auroc * 100, 1)), str(np.round(image_ap * 100, 1))]
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)

        elif args.metrics == "pixel-level":
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table += [str(np.round(pixel_auroc * 100, 1)), str(np.round(pixel_aupro * 100, 1)),
                      str(np.round(pixel_ap * 100, 1))]
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_ap_list.append(pixel_ap)

        elif args.metrics == "image-pixel-level":
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table += [
                str(np.round(pixel_auroc * 100, 1)),
                str(np.round(pixel_aupro * 100, 1)),
                str(np.round(pixel_ap * 100, 1)),
                str(np.round(image_auroc * 100, 1)),
                str(np.round(image_ap * 100, 1)),
            ]
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_ap_list.append(pixel_ap)
        else:
            raise ValueError(f"Unknown metrics mode: {args.metrics}")

        table_ls.append(table)

    if args.metrics == "image-level":
        table_ls.append(["mean", str(np.round(np.mean(image_auroc_list) * 100, 1)),
                         str(np.round(np.mean(image_ap_list) * 100, 1))])
        output = tabulate(table_ls, headers=["objects", "image_auroc", "image_ap"], tablefmt="pipe")
    elif args.metrics == "pixel-level":
        table_ls.append(["mean", str(np.round(np.mean(pixel_auroc_list) * 100, 1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, 1)),
                         str(np.round(np.nanmean(pixel_ap_list) * 100, 1))])
        output = tabulate(table_ls, headers=["objects", "pixel_auroc", "pixel_aupro", "pixel_ap"], tablefmt="pipe")
    else:
        table_ls.append(["mean", str(np.round(np.mean(pixel_auroc_list) * 100, 1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, 1)),
                         str(np.round(np.nanmean(pixel_ap_list) * 100, 1)),
                         str(np.round(np.mean(image_auroc_list) * 100, 1)),
                         str(np.round(np.mean(image_ap_list) * 100, 1))])
        output = tabulate(table_ls,
                         headers=["objects", "pixel_auroc", "pixel_aupro", "pixel_ap", "image_auroc", "image_ap"],
                         tablefmt="pipe")

    logger.info("\n%s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PACE-CLIP training Test", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/visa")
    parser.add_argument("--save_path", type=str, default="./results/")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--metrics", type=str, default="image-pixel-level")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--sigma", type=int, default=4)

    parser.add_argument("--gamma_ca", type=float, default=0.1)
    parser.add_argument("--lambda_router", type=float, default=0.7)
    parser.add_argument("--topk_ratio", type=float, default=0.005)
    parser.add_argument("--cpg_topk_ratio", type=float, default=0.05)
    parser.add_argument("--patch_router_tau", type=float, default=0.1)
    parser.add_argument("--pixel_patch_ratio", type=float, default=0.25)
    parser.add_argument("--image_patch_ratio", type=float, default=0.25)
    parser.add_argument("--fusion_floor_ratio", type=float, default=0.10)
    parser.add_argument("--fusion_temperature", type=float, default=1.0)

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
