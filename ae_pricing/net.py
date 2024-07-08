from typing import Optional, Sequence, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn



class Net(nn.Module):
    @dataclass
    class Config:
        pass


    def __init__(self):
        super().__init__()
        self.config = self.Config()




class MeanEnsemble(nn.Module):
    def __init__(self, nets: list[nn.Module]):
        super().__init__()
        self.nets = nn.ModuleList(nets)
    
    def forward(self, *args):
        # handle multiple inputs/outputs
        raws = [net(*args) for net in self.nets]
        
        if isinstance(raws[0], tuple):
            raws = list(zip(*raws))
            raws = [torch.stack(o, dim=0) for o in raws]
            outputs = [o.mean(dim=0) for o in raws]
        else:
            raws = torch.stack(raws, dim=0)
            outputs = raws.mean(dim=0)

        return outputs, raws



class GKXModel(nn.Module):
    def __init__(self, beta_arch: Optional[Sequence[int]]=None, factor_arch: Optional[Sequence[int]]=None, 
                 num_chrs: int=95, num_factors: int=6) -> None:
        super().__init__()
        self.beta_model = BetaModel(beta_arch, num_chrs, num_factors)
        self.factor_model = FactorModel(factor_arch, num_chrs, num_factors)


    def forward(self, chrs: torch.Tensor, rets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dimension Legends
        N: Assets, P: Characteristics, K: Factors
        
        params:
            chrs: (N, P)-shaped 2D tensor, containing asset (firm) characteristics.
            rets: (N, 1)-shaped 2D tensor, containing monthly return of each asset(firm)s.
        """
        port_rets = torch.matmul(
            torch.inverse(torch.matmul(chrs.T, chrs)), torch.matmul(chrs.T, rets)
        ).squeeze()  # (P,)-shaped 1D tensor, Characteristic-managed portfolio returns
        port_rets = torch.where(port_rets.abs() > 0.5, torch.zeros_like(port_rets), port_rets)

        factors, factor_intermediates = self.factor_model.forward(port_rets)  # (K,)-shaped 1D tensor - Factors estimated by port returns.
        betas = self.beta_model.forward(chrs)  # (N, K)-shaped 2D tensor - Betas, asset exposures to factors.

        # (N, K) @ (1, K) -> (N, K) sum-> (N, 1).
        # predicted individual asset returns.
        pred_rets = (betas * factors.unsqueeze(0)).sum(dim=1, keepdim=True)

        return pred_rets, betas, factors, port_rets, factor_intermediates



class BetaModel(nn.Module):
    def __init__(self, arch: Optional[Sequence[int]]=None, num_chrs: int=95, num_factors: int=6) -> None:
        super().__init__()
        if arch is None:
            arch = [32, 16, 8]
        self.net = construct_mlp([num_chrs,] + arch + [num_factors,], nn.SiLU, do_norm=True, bias=False)
    

    def forward(self, chrs: torch.Tensor) -> torch.Tensor:  # (..., P) -> (..., K)
        """
        """
        return self.net(chrs)



class FactorModel(nn.Module):
    def __init__(self, arch: Optional[Sequence[int]]=None, num_chrs: int=95, num_factors: int=6) -> None:
        super().__init__()
        if arch is None:
            arch = [32,]
        self.net = construct_mlp(
            [num_chrs,] + arch + [num_factors,], 
            nn.SiLU, do_norm=False, bias=False, return_intermediates=True)
    

    def forward(self, returns: torch.Tensor) -> torch.Tensor:  # (..., P) -> (..., K)
        """
        """
        return self.net(returns)


class MLP(nn.Module):
    def __init__(self, arch: Sequence[int], activation: torch.nn.Module, do_norm: bool=False,
                  norm_cls: torch.nn.Module=nn.BatchNorm1d, bias: bool = True,
                  last_activation: torch.nn.Module=nn.Identity, return_intermediates: bool = False):
        super().__init__()

        layers = []
        self.return_lasthidden = return_intermediates

        for i, (inp, out) in enumerate(zip(arch[:-1], arch[1:])):
            layers.append(nn.Linear(inp, out, bias=bias))
            if i < len(arch) - 1:
                layers.append(activation())
                if do_norm:
                    layers.append(norm_cls(out))
        layers.append(last_activation())

        self.layers = nn.ModuleList(layers)

    
    def forward(self, x):
        # bullshit
        if self.return_lasthidden:
            for layer in self.layers[:-1]:
                x = layer(x)
            out = self.layers[-1](x)
            return out, x
        else:
            for layer in self.layers:
                x = layer(x)
            return x


def construct_mlp(arch: Sequence[int], activation: torch.nn.Module, do_norm: bool=False,
                  norm_cls: torch.nn.Module=nn.BatchNorm1d, bias: bool = True,
                  last_activation: torch.nn.Module=nn.Identity, return_intermediates: bool = False) -> torch.nn.Module:
    """
    Constructs a Sequential MLP with given architecture and activation.

    params:
        - arch: A sequence of int, which is the hidden sizes of the MLP.
                Input and output dimension should be also included in the arch.
        - activation: An activation layer class.
        - do_norm: whether to use normalization layers or not.
        - norm_cls: normalization layer class
        - last_activation: Activation on top of the network. default: nn.Identity

    returns:
        - A torch.nn.Sequential MLP model
    """

    return MLP(arch, activation, do_norm, norm_cls, bias, last_activation, return_intermediates)
    """layers = []

    for inp, out in zip(arch[:-1], arch[1:]):
        layers.append(nn.Linear(inp, out, bias=bias))
        layers.append(activation())
        if do_norm:
            layers.append(norm_cls(out))
    if do_norm:
        layers.pop(-1)  # removes last normalization
    layers.pop(-1)  # removes last activation
    layers.append(last_activation())

    return nn.Sequential(*layers)"""