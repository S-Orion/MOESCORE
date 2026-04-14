import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.2, beta: float = 0.9999, cbl: bool = False):
        super().__init__()
        self.margin = margin
        self.beta = beta
        self.cbl = cbl

    def forward(self, pred_score, gt_score, num_class):
        if pred_score.dim() > 2:
            pred_score = pred_score.mean(dim=1).contiguous().view(-1)

        gt_diff = gt_score.unsqueeze(1) - gt_score.unsqueeze(0)
        pred_diff = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)

        # margin-based loss
        loss = torch.maximum(
            torch.zeros_like(gt_diff),
            torch.abs(pred_diff - gt_diff) - self.margin
        )

        if self.cbl:
            weights = (1 - self.beta) / (1 - torch.pow(self.beta, num_class))
            weights = weights / torch.sum(weights)
            class_mat = torch.outer(weights, weights).sqrt()
            loss = loss * class_mat
        
        return loss.mean() / 2



class ClippedMSELoss(nn.Module):
    def __init__(self, criterion, tau: float = 0.5, mode='utt', beta: float = 0.9999, cbl: bool = False):
        super().__init__()
        self.tau = torch.tensor(tau, dtype=torch.float)
        self.criterion = criterion
        self.mode = mode
        self.beta = beta
        self.cbl = cbl

    def forward(self, pred_score, gt_score, num_class):
        time = pred_score.shape[1] if pred_score.dim() > 2 else 1

        if self.mode == 'utt':
            pred_score = pred_score.mean(dim=1) if pred_score.dim() > 2 else pred_score
        else:
            gt_score = gt_score.unsqueeze(1).repeat(1, time)

        pred_score = pred_score.contiguous().view(-1)
        loss = self.criterion(pred_score, gt_score)

        threshold = torch.abs(pred_score - gt_score) > self.tau

        if self.cbl:
            num_class = num_class.unsqueeze(1)
            weights = (1 - self.beta) / (1 - torch.pow(self.beta, num_class))
            weights = weights / torch.sum(weights)
            loss = loss * weights

        return (threshold * loss).mean()



class CombineLosses(nn.Module):
    def __init__(self, loss_weights: list, loss_instances: list):
        super().__init__()
        self.loss_weights = loss_weights
        self.loss_instances = nn.ModuleList(loss_instances)

    def forward(self, pred_score, gt_score, num_class):
        loss = torch.tensor(0.0, dtype=torch.float, device=pred_score.device)
        for w, loss_fn in zip(self.loss_weights, self.loss_instances):
            loss += w * loss_fn(pred_score, gt_score, num_class)
        return loss


def get_loss_function(loss_cfg: dict) -> nn.Module:
    loss_weights = loss_cfg["loss_weights"]

    clipped_mse = ClippedMSELoss(
        criterion=nn.MSELoss(reduction="none"),
        tau=loss_cfg["clipped_mse"]["tau"],
        mode=loss_cfg["clipped_mse"]["mode"],
        beta=loss_cfg["clipped_mse"]["beta"],
        cbl=loss_cfg["clipped_mse"]["cbl"]
    )

    contrastive = ContrastiveLoss(
        margin=loss_cfg["contrastive"]["margin"],
        beta=loss_cfg["contrastive"]["beta"],
        cbl=loss_cfg["contrastive"]["cbl"]
    )


    return CombineLosses(
        loss_weights=loss_weights,
        loss_instances=[clipped_mse, contrastive]

    )