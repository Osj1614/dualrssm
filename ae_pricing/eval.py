from typing import Sequence, Any
import torch
import numpy as np
from sklearn.metrics import r2_score
from ae_pricing.net import GKXModel
from ae_pricing.data import PandasDataLoader, GKXDataLoader


@torch.no_grad()
def total_r2_score(model: GKXModel, dloader: PandasDataLoader, indices: Sequence[int], device: Any="cpu") -> float:
    """
    Computes total R2 score of the model on the given data loader.
    """
    model = model.eval().to(device)

    ret_squareds = 0.0  # sum of squared returns for each timepoint.
    pred_squareds = 0.0  # sum of squared prediction errors for each timepoint.

    for month_idx in indices:
        date, chrs, rets = dloader[month_idx]
        chrs, rets = torch.from_numpy(chrs).float().to(device, non_blocking=True), torch.from_numpy(rets).float().to(device, non_blocking=True)

        ret_squareds += rets.square().sum().item()
        pred_rets, _, _, _ = model(chrs, rets)
        pred_squareds += (pred_rets - rets).square().sum().item()

    return 1 - pred_squareds / ret_squareds


@torch.no_grad()
def pred_r2_score(model: GKXModel, dloader: GKXDataLoader, train_indices: Sequence[int],
                   target_indices: Sequence[int], device: Any="cpu") -> float:
    """
    Computes R2 score of the model on the given data loader.
    Indices should be aligned chronologically, to correctly calculate the predictive factors.
    """
    model = model.eval().to(device)

    ret_squareds = 0.0
    pred_squareds = 0.0
    
    # calculate factors first
    factors = []
    for date in train_indices:
        chrs, rets = dloader.chrs[date], dloader.rets[date]

        chrs, rets = torch.from_numpy(chrs).float().to(device, non_blocking=True), torch.from_numpy(rets).float().to(device, non_blocking=True)
        (_, _, factor, _, _), _ = model(chrs, rets)
        factors.append(factor)

    predictive_factor = torch.stack(factors, dim=0).mean(dim=0, keepdim=True)

    for date in target_indices:
        chrs, rets = dloader.chrs[date], dloader.rets[date]
        chrs, rets = torch.from_numpy(chrs).float().to(device, non_blocking=True), torch.from_numpy(rets).float().to(device, non_blocking=True)

        ret_squareds += rets.square().sum().item()
        (_, betas, true_factor, _, _), _ = model(chrs, rets)

        pred_rets = (betas * predictive_factor).sum(dim=1, keepdim=True)
        pred_squareds += (pred_rets - rets).square().sum().item()

    return 1 - pred_squareds / ret_squareds