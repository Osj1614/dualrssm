import numpy as np


def rank_ic(rets: np.ndarray, preds: np.ndarray) -> float:
    """
    Computes the rank IC of a model's predictions.
    
    Parameters
    ----------
    rets : np.ndarray
        Array of returns, Shape (Timesteps, Assets)
    preds : np.ndarray
        Array of predictions, Shape (Timesteps, Assets)
    
    Returns
    -------
    float
        Rank IC.
    """
    ranks = np.argsort(rets, axis=-1)
    pred_ranks = np.argsort(preds, axis=-1)
    
    return (
        (ranks - ranks.mean(axis=-1, keepdims=True)) * (pred_ranks - pred_ranks.mean(axis=-1, keepdims=True))
        / (ranks.std(axis=-1, keepdims=True) * pred_ranks.std(axis=-1, keepdims=True))
    ).mean()