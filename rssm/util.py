from typing import Any
import torch


def extract_scalars(d: dict[str, Any]):
    ret = {}
    for k, v in d.items():
        if type(v) is torch.Tensor and v.ndim == 0:
            ret[k] = v.item()
        elif type(v) is float or type(v) is int:
            ret[k] = v
    return ret


def mean_dict(d: dict[str, list]) -> dict[str, float]:
    return {k: sum(v) / len(v) for k, v in d.items()}