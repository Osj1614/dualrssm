from typing import Sequence, Any
from dataclasses import dataclass, field
from copy import deepcopy

import torch
import torch.nn as nn

from rssm.distribution import MultiCategoricalWithLogits


class MLP(nn.Module):
    @dataclass
    class Config:
        arch: list[int] = field(default_factory=lambda: [128, 128])
          
        activation: str = "Tanh"
        norm_cls: str = "LayerNorm"
        bias: bool = True
        last_activation: str = "Identity"

        def __post_init__(self):
            # TODO: nn.Module encode pyrallis register
            if type(self.activation) is str:
                self.activation: nn.Module = getattr(nn, self.activation)
            if type(self.norm_cls) is str:
                self.norm_cls: nn.Module = getattr(nn, self.norm_cls)
            if type(self.last_activation) is str:
                self.last_activation: nn.Module = getattr(nn, self.last_activation)


    def __init__(self, config: Config, **kwargs):
        super().__init__()
        self.config = config

        layers = []

        for i, (inp, out) in enumerate(zip(config.arch[:-1], config.arch[1:])):
            layers.append(nn.Linear(inp, out, bias=config.bias))
            if i < len(config.arch) - 2:
                layers.append(config.activation())
                if config.norm_cls is not None:
                    layers.append(config.norm_cls(out))
        if config.last_activation is not None:
            layers.append(config.last_activation())

        self.layers = nn.ModuleList(layers)

    
    def forward(self, x):
        outs = [x,]
        for layer in self.layers:
            outs.append(layer(outs[-1]))

        return outs[-1], outs[:-1]



class FactorModel(nn.Module):
    @dataclass
    class Config:
        chr_dim: int = 94
        stock_deter_dim: int = 64
        stock_stoch_dim: int = 128
        market_deter_dim: int = 64
        market_stoch_dim: int = 128
        stock_aggr_dim: int = 128  # OSJ GAURANTEE
        factor_dim: int = 32
        use_alpha: bool = True

        stock_prior_net_cfg: MLP.Config = field(default_factory=MLP.Config)
        stock_post_net_cfg: MLP.Config = field(default_factory=MLP.Config)
        market_prior_net_cfg: MLP.Config = field(default_factory=MLP.Config)
        market_post_net_cfg: MLP.Config = field(default_factory=MLP.Config)
        #aggr_net_cfg: MLP.Config = field(default_factory=MLP.Config)
        factor_head_cfg: MLP.Config = field(default_factory=MLP.Config)
        beta_head_cfg: MLP.Config = field(default_factory=MLP.Config)
        alpha_head_cfg: MLP.Config = field(default_factory=MLP.Config)
        stock_aggr_net_cfg: MLP.Config = field(default_factory=lambda: MLP.Config(arch=[256, 256]))
        recon_head_cfg: MLP.Config = field(default_factory=MLP.Config)

        stock_kl_alpha: float = 0.9
        #stock_dist_cls: str = 'Normal'
        #stock_dist_postops: str = 'Identity,Softplus'
        #stock_dist_n_params: int = 2
        stock_dist_cls: str = "MultiCategoricalWithLogits"
        stock_dist_postops: str = "Identity"
        stock_dist_n_params: int = 1
        stock_dist_n_classes: int = 16  # Ignored if not Categorical

        market_kl_alpha: float = 0.9
        #market_dist_cls: str = 'Normal'
        #market_dist_postops: str = 'Identity,Softplus'
        #market_dist_n_params: int = 2
        market_dist_cls: str = "MultiCategoricalWithLogits"
        market_dist_postops: str = "Identity"
        market_dist_n_params: int = 1
        market_dist_n_classes: int = 16  # Ignored if not Categorical

        loss_weights: dict[str, float] = field(default_factory=lambda: {
            'ret_pred_loss': 1.0,
            'alpha_loss': 1.0,
            'stock_kl_loss': 16.0,
            'market_kl_loss': 16.0,
            #'chr_recon_loss': 0.5,
            #'beta_ortho_loss': 1e-6
        })
        

        def __post_init__(self):
            self.stock_prior_net_cfg.arch = [self.stock_deter_dim,] + self.stock_prior_net_cfg.arch + [self.stock_stoch_dim * self.stock_dist_n_params,]
            self.stock_post_net_cfg.arch = [self.stock_deter_dim + self.chr_dim,] + self.stock_post_net_cfg.arch + [self.stock_stoch_dim * self.stock_dist_n_params,]
            self.market_prior_net_cfg.arch = [self.market_deter_dim,] + self.market_prior_net_cfg.arch + [self.market_stoch_dim * self.market_dist_n_params,]
            self.market_post_net_cfg.arch = [self.market_deter_dim + self.stock_aggr_dim,] + self.market_post_net_cfg.arch + [self.market_stoch_dim * self.market_dist_n_params,]
            self.stock_aggr_net_cfg.arch = [self.chr_dim + 1,] + self.stock_aggr_net_cfg.arch + [self.stock_aggr_dim,]
            
            self.factor_head_cfg.arch = [self.market_deter_dim + self.market_stoch_dim,] + self.factor_head_cfg.arch + [self.factor_dim,]
            self.beta_head_cfg.arch = [self.stock_stoch_dim + self.stock_deter_dim,] + self.beta_head_cfg.arch + [self.factor_dim,]
            self.alpha_head_cfg.arch = [self.stock_stoch_dim + self.stock_deter_dim,] + self.alpha_head_cfg.arch + [1,]
            
            if 'chr_recon_loss' in self.loss_weights and self.loss_weights['chr_recon_loss'] != 0.0:
                self.recon_head_cfg.arch = [self.stock_stoch_dim + self.stock_deter_dim,] + self.recon_head_cfg.arch + [self.chr_dim,]


    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        
        # Distribution
        try:
            self.stock_dist_cls = getattr(torch.distributions, self.config.stock_dist_cls)
        except:
            self.stock_dist_cls = eval(self.config.stock_dist_cls)

        try:
            self.market_dist_cls = getattr(torch.distributions, self.config.market_dist_cls)
        except:
            self.market_dist_cls = eval(self.config.market_dist_cls)

        self.stock_dist_postops = [getattr(torch.nn, op)() for op in self.config.stock_dist_postops.split(',')]
        self.market_dist_postops = [getattr(torch.nn, op)() for op in self.config.market_dist_postops.split(',')]
        
        
        # Recurrent
        self.market_gru_hidden = nn.Parameter(torch.zeros(1, 1, self.config.market_deter_dim))  # (1, 1, MD)
        self.stock_gru_hidden = nn.Parameter(torch.zeros(1, 1, self.config.stock_deter_dim)) # (1, 1, SD)
        self.market_gru = nn.GRU(
            input_size=self.config.market_stoch_dim,  # SS
            hidden_size=self.config.market_deter_dim, num_layers=1, batch_first=True # SD
        )
        self.stock_gru = nn.GRU(
            input_size=self.config.stock_stoch_dim + self.config.market_deter_dim + self.config.market_stoch_dim,  # CLS, SD, SS
            hidden_size=self.config.stock_deter_dim, num_layers=1, batch_first=True  # CLD
        )
       
        # Distribution Networks
        self.stock_prior_net = MLP(self.config.stock_prior_net_cfg)
        self.stock_post_net = MLP(self.config.stock_post_net_cfg)
        self.market_prior_net = MLP(self.config.market_prior_net_cfg)
        self.market_post_net = MLP(self.config.market_post_net_cfg)

        # Aggregators
        #self.stoch_aggr = NonLinearAverage(MLP.Config(self.config.aggr_net_cfg, nn.ReLU, nn.LayerNorm))
        #self.deter_aggr = NonLinearAverage(MLP.Config(self.config.aggr_net_cfg, nn.ReLU, nn.LayerNorm))
        self.stock_aggr = AttentionStockAggr(self.config.stock_aggr_net_cfg)
        
        # Heads
        self.factor_net = MLP(self.config.factor_head_cfg)
        self.beta_net = MLP(self.config.beta_head_cfg)
        if self.config.use_alpha:
            self.alpha_net = MLP(self.config.alpha_head_cfg)
        if 'chr_recon_loss' in self.config.loss_weights and self.config.loss_weights['chr_recon_loss'] != 0.0:
            self.recon_net = MLP(self.config.recon_head_cfg)


    def forward(self, chrs: torch.Tensor, rets: torch.Tensor, do_sample: bool = True) -> dict[str, Any]:
        # chrs: (B, T, S, C)
        # rets: (B, T, S, 1)
        batch_size, seq_len, stock_size, chr_dim = chrs.shape
        B, T, S, C, MD, SD, MS, SS = batch_size, seq_len, stock_size, chr_dim, self.config.market_deter_dim, self.config.stock_deter_dim, self.config.market_stoch_dim, self.config.stock_stoch_dim
        
        market_hiddens = [self.market_gru_hidden.repeat(1, batch_size, 1)]  # (1, B, M)
        stock_hiddens = [self.stock_gru_hidden.repeat(1, batch_size * stock_size, 1)]  # (1, B*S, SD)
        factors = []
        betas = []
        alphas = []
        stock_prior_params = []
        stock_posterior_params = []
        stock_posterior_samples = []
        market_prior_params = []
        market_posterior_params = []
        market_posterior_samples = []
        stock_aggrs = []

        for t in range(seq_len):
            # Stock Aggregation
            stock_aggr = self.stock_aggr(torch.cat((chrs[:, t], rets[:, t]), dim=-1), keepdims=True)  # (B, S, C+1) -> (B, 1, A)
            assert stock_aggr.shape == (B, 1, self.config.stock_aggr_dim)
            stock_aggrs.append(stock_aggr)
            
            # Prior Posterior
            stock_posterior_param = self.stock_post_net(torch.cat([chrs[:, t], stock_hiddens[-1].view(B, S, SD)], dim=-1))[0].chunk(self.config.stock_dist_n_params, dim=-1)
            assert stock_posterior_param[0].shape == (B, S, SS)
            stock_prior_param = self.stock_prior_net(stock_hiddens[-1].view(B, S, SD))[0].chunk(self.config.stock_dist_n_params, dim=-1)
            assert stock_prior_param[0].shape == (B, S, SS)

            market_posterior_param = self.market_post_net(torch.cat([stock_aggrs[-1], market_hiddens[-1][0].unsqueeze(1),], dim=-1))[0].chunk(self.config.market_dist_n_params, dim=-1)
            assert market_posterior_param[0].shape == (B, 1, MS)
            market_prior_param = self.market_prior_net(market_hiddens[-1][0].unsqueeze(1))[0].chunk(self.config.market_dist_n_params, dim=-1)
            assert market_prior_param[0].shape == (B, 1, MS)
            
            # Postops for parameters
            stock_posterior_param = tuple(op(param) for op, param in zip(self.stock_dist_postops, stock_posterior_param))  # (B, S, SD*D)
            stock_prior_param = tuple(op(param) for op, param in zip(self.stock_dist_postops, stock_prior_param))  # (B, S, SD*D)
            market_posterior_param = tuple(op(param) for op, param in zip(self.market_dist_postops, market_posterior_param))
            market_prior_param = tuple(op(param) for op, param in zip(self.market_dist_postops, market_prior_param))
            
            stock_posterior_params.append(stock_posterior_param)
            stock_prior_params.append(stock_prior_param)
            market_posterior_params.append(market_posterior_param)
            market_prior_params.append(market_prior_param)
            
            stock_posterior = self.stock_dist_cls(*stock_posterior_params[-1])
            if do_sample:
                stock_posterior_sample = stock_posterior.rsample()  # (B, S, SS)
            else:
                stock_posterior_sample = stock_posterior.mean
            assert stock_posterior_sample.shape == (B, S, SS)
            stock_posterior_samples.append(stock_posterior_sample)
            
            market_posterior = self.market_dist_cls(*market_posterior_params[-1])
            if do_sample:
                market_posterior_sample = market_posterior.rsample()  # (B, MS)
            else:
                market_posterior_sample = market_posterior.mean
            assert market_posterior_sample.shape == (B, 1, MS)
            market_posterior_samples.append(market_posterior_sample)

            # Factor and Beta
            beta = self.beta_net(torch.cat([stock_hiddens[-1].view(B, S, SD), stock_posterior_sample], dim=-1))[0]  # (B, S, 2H) -> (B, S, F)
            assert beta.shape == (B, S, self.config.factor_dim)
            betas.append(beta)
            if self.config.use_alpha:
                alpha = self.alpha_net(torch.cat([stock_hiddens[-1].view(B, S, SD), stock_posterior_sample], dim=-1))[0]  # (B, S, 2H) -> (B, S, 1)
                assert alpha.shape == (B, S, 1)
                alphas.append(alpha)
            
            # Recurrent
            market_hiddens.append(
                self.market_gru(market_posterior_sample, market_hiddens[-1])[1]
            )  # GRU( (B, 1, MS), (1, B, MD) ) -> _, (1, B, MD)
            
            factor = self.factor_net(torch.cat([market_hiddens[-1][0], market_posterior_sample[:, 0]], dim=-1))[0]  # (B, F)
            assert factor.shape == (B, self.config.factor_dim)
            factors.append(factor)
            
            stock_hiddens.append(
                self.stock_gru(
                    torch.cat(
                        [
                            market_hiddens[-2].reshape(B, 1, MD).expand(-1, S, -1),  # (B, S, MD)
                            market_posterior_sample.expand(-1, S, -1),  # (B, S, MS)
                            stock_posterior_sample  # (B, S, SS)
                        ], dim=-1
                    ).view(B*S, 1, MD + MS + SS),
                    stock_hiddens[-1]
                )[1]
            )  #
        factors = torch.stack(factors, dim=1)  
        betas = torch.stack(betas, dim=1)
        if self.config.use_alpha:
            alphas = torch.stack(alphas, dim=1).squeeze(-1)
        else:
            alphas = 0.0
        stock_posterior_samples = torch.stack(stock_posterior_samples, dim=1)
        market_posterior_samples = torch.stack(market_posterior_samples, dim=1)
        
        return {
            'factors': factors,
            'betas': betas,
            'alphas': alphas,
            'stock_prior_params': stock_prior_params,
            'stock_posterior_params': stock_posterior_params,
            'stock_hiddens': stock_hiddens,
            'stock_posterior_samples': stock_posterior_samples,
            'market_prior_params': market_prior_params,
            'market_posterior_params': market_posterior_params,
            'market_hiddens': market_hiddens,
            'market_posterior_samples': market_posterior_samples,
            'stock_aggrs': stock_aggrs,
        }


    def loss(self, chrs: torch.Tensor, rets: torch.Tensor, do_sample: bool=True) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        # chrs: (B, T, S, C)
        # rets: (B, T, S, 1)
        batch_size, seq_len, stock_size, chr_dim = chrs.shape
        B, T, S, C, MD, SD, MS, SS = batch_size, seq_len, stock_size, chr_dim, self.config.market_deter_dim, self.config.stock_deter_dim, self.config.market_stoch_dim, self.config.stock_stoch_dim
        
        out = self.forward(chrs, rets, do_sample)
        betas, factors, alphas = out['betas'], out['factors'], out['alphas']
        stock_prior_params, stock_posterior_params = out['stock_prior_params'], out['stock_posterior_params']
        
        # Prediction Loss
        pred_rets = torch.einsum('btsf,btf->bts', betas, factors) # (B, T, S)
        assert pred_rets.shape == (B, T, S)
        ret_pred_loss = torch.nn.functional.mse_loss(pred_rets, rets.squeeze(-1))
        
        # Residual (Alpha) Loss
        if self.config.use_alpha:
            alpha_loss = torch.nn.functional.mse_loss(alphas, rets.squeeze(-1) - pred_rets)
        else:
            alpha_loss = 0.0
        
        # KL Loss
        stock_kl_loss = 0.0
        for stock_prior_param, stock_posterior_param in zip(stock_prior_params, stock_posterior_params):
            stock_kl_loss += (
                self.config.stock_kl_alpha * torch.distributions.kl.kl_divergence(
                    self.stock_dist_cls(*tuple(p.detach() for p in stock_posterior_param)), self.stock_dist_cls(*stock_prior_param)
                ).mean()
                + (1 - self.config.stock_kl_alpha) * torch.distributions.kl.kl_divergence(
                    self.stock_dist_cls(*stock_posterior_param), self.stock_dist_cls(*tuple(p.detach() for p in stock_prior_param))
                ).mean()
            )
        stock_kl_loss = stock_kl_loss / len(stock_prior_params)

        market_kl_loss = 0.0
        for market_prior_param, market_posterior_param in zip(out['market_prior_params'], out['market_posterior_params']):
            market_kl_loss += (
                self.config.market_kl_alpha * torch.distributions.kl.kl_divergence(
                    self.market_dist_cls(*tuple(p.detach() for p in market_posterior_param)), self.market_dist_cls(*market_prior_param)
                ).mean()
                + (1 - self.config.market_kl_alpha) * torch.distributions.kl.kl_divergence(
                    self.market_dist_cls(*market_posterior_param), self.market_dist_cls(*tuple(p.detach() for p in market_prior_param))
                ).mean()
            )
        
        if 'chr_recon_loss' in self.config.loss_weights and self.config.loss_weights['chr_recon_loss'] != 0.0:
            out['chr_recon_loss'] = torch.nn.functional.mse_loss(chrs, self.recon_net(
                torch.cat([
                    torch.stack([s.view(B, S, SD) for s in out['stock_hiddens'][:-1]], dim=1),
                    out['stock_posterior_samples']
                ], dim=-1))[0]
            )
        
        if 'beta_ortho_loss' in self.config.loss_weights and self.config.loss_weights['beta_ortho_loss'] != 0.0:
            # betas: (B, T, S, F)
            # betas should be orthogonal along stock dimension
            betas_ = betas.flatten(0, 1)  # (B*T, S, F)
            betabeta = torch.bmm(betas_.transpose(1, 2), betas_)  # (B*T, F, F)
            betaoffdiag = betabeta * (1 - torch.eye(self.config.factor_dim, device=betabeta.device).unsqueeze(0))
            
            betadiag = betabeta * torch.eye(self.config.factor_dim, device=betabeta.device).unsqueeze(0)
            out['beta_ortho_loss'] = betaoffdiag.square().mean() - betadiag.square().mean()
        
        out['posterior_return'] = pred_rets
        out['ret_pred_loss'] = ret_pred_loss
        out['alpha_loss'] = alpha_loss
        out['stock_kl_loss'] = stock_kl_loss
        out['market_kl_loss'] = market_kl_loss
        out['loss'] = sum(out[name] * self.config.loss_weights[name] for name in self.config.loss_weights if name in out)
        return out


    def predict(self, chrs: torch.Tensor, rets: torch.Tensor, do_sample: bool = True) -> dict[str, Any]:
        # chrs: (B, T, S, C)
        # rets: (B, T, S, 1)
        out = self.loss(chrs, rets, do_sample)
        batch_size, seq_len, stock_size, chr_dim = chrs.shape
        B, T, S, C, MD, SD, MS, SS = batch_size, seq_len, stock_size, chr_dim, self.config.market_deter_dim, self.config.stock_deter_dim, self.config.market_stoch_dim, self.config.stock_stoch_dim
        
        stock_prior = self.stock_dist_cls(*out['stock_prior_params'][-1])
        if do_sample:
            stock_prior_sample = stock_prior.sample()
        else:
            stock_prior_sample = stock_prior.mean
        
        market_prior = self.market_dist_cls(*out['market_prior_params'][-1])
        if do_sample:
            market_prior_sample = market_prior.sample()
        else:
            market_prior_sample = market_prior.mean
        
        # OSJ TOUCH: GRU OUTPUT [0] GOOD
        prior_market_latent = self.market_gru(market_prior_sample, out['market_hiddens'][-2])[1][0]  # [(B,1,H), (B,1,H)] -> _, (1, B, M) -> (B, M) 

        prior_factor = self.factor_net(torch.cat([prior_market_latent, market_prior_sample[:, 0]], dim=-1))[0]  # (B, F)
        prior_beta = self.beta_net(torch.cat([out['stock_hiddens'][-2].view(B, S, SD), stock_prior_sample], dim=-1))[0]  # (B, S, F)
        if self.config.use_alpha:
            prior_alpha = self.alpha_net(torch.cat([out['stock_hiddens'][-2].view(B, S, SD), stock_prior_sample], dim=-1))[0].squeeze(-1)  # (B, S)
        else:
            prior_alpha = 0.0

        pred_return = torch.einsum('bsf,bf->bs', prior_beta, prior_factor)  # (B, S)
        ret_pred_loss = torch.nn.functional.mse_loss(pred_return, rets[:, -1, :, 0])
        
        pred_r2 = 1 - (pred_return - rets[:, -1, :, 0]).square().sum() / (rets[:, -1, :, 0]).square().sum()
        
        posterior_return = out['posterior_return'][:, -1]  # (B, S)
        total_r2 = 1 - (posterior_return - rets[:, -1, :, 0]).square().sum() / (rets[:, -1, :, 0]).square().sum()
        
        if self.config.use_alpha:
            out['prior_alpha'] = prior_alpha
            out['posterior_alpha'] = out['alphas'][:, -1]
            out['alpha_pred_r2'] = 1 - (pred_return - (rets[:, -1, :, 0] + prior_alpha)).square().sum() / (rets[:, -1, :, 0]).square().sum()
            out['alpha_total_r2'] = 1 - (posterior_return - (rets[:, -1, :, 0] + out['alphas'][:, -1])).square().sum() / (rets[:, -1, :, 0]).square().sum()
            
        
        out.update({
            'pred_return': pred_return,
            'prior_ret_pred_loss': ret_pred_loss,
            'prior_factor': prior_factor,
            'prior_beta': prior_beta,
            'total_r2': total_r2,
            'pred_r2': pred_r2,
        })
        
        return out



class NonLinearAverage(nn.Module):
    def __init__(self, mlp_config: MLP.Config):
        super().__init__()
        self.mlp = MLP(mlp_config)
    

    def forward(self, x):
        # x: (..., K, D)
        # return: (..., D)
        return self.mlp(x)[0].mean(dim=-2)



#class MultiDiscreteSeparator(nn.Module):
#    def __init__(self, inp_dim:int = 256, chunk_dim=16):
#        self.inp_dim = inp_dim
#        self.chunk_dim = chunk_dim

class AttentionStockAggr(nn.Module):
    def __init__(self, enc_mlp_config: MLP.Config):
        super().__init__()
        self.enc_mlp = MLP(enc_mlp_config)
        self.out_dim = enc_mlp_config.arch[-1]  # D
        self.aggr_token = nn.Parameter(torch.zeros(1, 1, enc_mlp_config.arch[-1], requires_grad=True))
        self.attn = nn.MultiheadAttention(embed_dim=enc_mlp_config.arch[-1], num_heads=1, batch_first=True)
    

    def forward(self, chrs: torch.Tensor, keepdims: bool=False) -> torch.Tensor:  # (B, S, C) -> (B, C)
        B, S, C = chrs.shape
        
        embeds = self.enc_mlp(chrs)[0]  # (B, S, C) -> (B, S, D)
        embeds = self.attn(self.aggr_token.expand(B, 1, -1), embeds, embeds)[0]  # (B, 1, D), (B, S, D), (B, S, D) -> (B, 1, D)

        if not keepdims:
            embeds = embeds.view(B, self.out_dim)
        
        return embeds