"""Pixel-Anchored Balanced Evidence Fusion (PABEF)."""

import torch


class PABEFFusion(torch.nn.Module):
    """
    Image-level evidence fusion:
        evidence = [global_score, pixel_L1_topk, pixel_L2_topk, pixel_L3_topk, pixel_L4_topk]

    floor mixing prevents single-evidence collapse:
        w = (1-floor) * softmax(logits / T) + floor / num_evidence
    """
    def __init__(self, num_evidence=5, temperature=1.0, floor_ratio=0.10):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(num_evidence))
        self.temperature = float(temperature)
        self.floor_ratio = float(floor_ratio)

    def get_weights(self):
        raw_w = torch.softmax(self.logits / self.temperature, dim=0)
        n = raw_w.numel()
        return (1.0 - self.floor_ratio) * raw_w + self.floor_ratio / n

    def forward(self, global_score, layer_topk_scores):
        evidences = torch.stack([global_score] + layer_topk_scores, dim=1)  # [B, 5]
        w = self.get_weights().to(evidences.device)
        image_score = (evidences * w.unsqueeze(0)).sum(dim=1)
        return torch.clamp(image_score, min=1e-6, max=1.0 - 1e-6), w

    def balance_loss(self):
        w = self.get_weights()
        mean = w.mean()
        var = ((w - mean) ** 2).mean()
        return var / (mean ** 2 + 1e-6)
