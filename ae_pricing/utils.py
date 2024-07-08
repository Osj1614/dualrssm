import numpy as np
import torch



def affix_dict(d: dict, prefix: str="", suffix: str="") -> dict:
    return {prefix + key + suffix: value for key, value in d.items()}


def tuple_arr_to_tensor(arrs, dtype, device):
    return tuple(torch.from_numpy(arr).to(dtype=dtype, device=device, non_blocking=True) for arr in arrs)


def l1_reg_loss(model: torch.nn.Module, factor: float=1e-3) -> torch.Tensor:
    """
    Computes L1 regularization loss of the model.
    """
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += param.abs().sum()
    return l1_loss * factor