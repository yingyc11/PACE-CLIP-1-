"""Dual-Granularity Decoupled Learning (DGDL) helpers."""

import torch


def shallow_deep_semantic_enhancement(patch_features, cross_attn, gamma_ca):
    shallow_feat = patch_features[0]
    deep_feat = patch_features[3].detach()
    fused_shallow_feat, _ = cross_attn(query=shallow_feat, key=deep_feat, value=deep_feat)
    optimized = list(patch_features)
    optimized[0] = patch_features[0] + gamma_ca * fused_shallow_feat
    return optimized
