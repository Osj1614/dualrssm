from dataclasses import dataclass, field, asdict
from typing import Any
from collections import defaultdict

import torch
from torch import nn
import wandb
import tqdm

from rssm.data import StockSeqDataLoader
from rssm.net import MLP, FactorModel
from rssm.util import extract_scalars, mean_dict



class SSFactorModelTrainer:
    @dataclass
    class Config:
        wandb_project: str = 'rssm_factor3'
        wandb_mode: str = 'online'
        histogram_names: list[str] = field(default_factory=lambda: ['betas', 'factors', 'posterior_return', 'pred_return'])
        
        device: str = 'cuda:0'
        n_epochs: int = 500
        
        optim_cls: str = 'Adam'
        optim_kwargs: dict = field(default_factory=lambda: {'lr': 3e-4})
        net_config: FactorModel.Config = field(default_factory=FactorModel.Config)
        dloader_config: StockSeqDataLoader.Config = field(default_factory=lambda: StockSeqDataLoader.Config(rank_return_same_time=True))
        train_start = '1985-01-01'
        val_start = '2010-01-01'
        test_start = '2012-01-01'
        test_end = '2019-01-01'


    def __init__(self, config: Config = None):
        if config is None:
            config = self.Config()
        self.config = config
        self.step = 0

        self.net = FactorModel(config.net_config).to(config.device)
        print(self.net)
        
        wandb.init(project=config.wandb_project, mode=config.wandb_mode, config=asdict(config))
        wandb.watch(self.net, log='all', log_graph=True, log_freq=10)  # Log gradients and parameters every 25 epochs
        self.dloader = StockSeqDataLoader(config.dloader_config)
        self.optim = getattr(torch.optim, config.optim_cls)(self.net.parameters(), **config.optim_kwargs)
    
    
    def log(self, metrics: dict[str, Any], prefix: str="", suffix: str="", incr_step: bool=True):
        metrics = {f'{prefix}/{k}_{suffix}': v for k, v in extract_scalars(metrics).items()}
        wandb.log(metrics, step=self.step)
        
        if incr_step:
            self.step += 1
    
    
    def save(self, path: str):
        torch.save(self.net.state_dict(), path + '.pth')
        torch.save(self.optim.state_dict(), path + '.optim')
    
    
    def load(self, path: str):
        self.net.load_state_dict(torch.load(path + '.pth'))
        self.optim.load_state_dict(torch.load(path + '.optim'))
    
    
    def train(self, epochs: int = 200):
        pbar = tqdm.trange(epochs)
        for epoch in pbar:
            self.net.train()
            
            train_metrics = defaultdict(list)
            for chrs, rets in self.dloader.get_split(self.config.train_start, self.config.val_start):
                chrs = torch.from_numpy(chrs).to(self.config.device, non_blocking=True)
                rets = torch.from_numpy(rets).to(self.config.device, non_blocking=True).unsqueeze(-1)
                
                self.optim.zero_grad()
                out = self.net.predict(chrs, rets)
                out['loss'].backward()
                self.optim.step()
                
                for k, v in extract_scalars(out).items():
                    train_metrics[k].append(v)
            train_metrics = mean_dict(train_metrics)
            self.log(train_metrics, prefix='train', suffix='mean')
            
            # Histogram of beta, factor, and predictions
            wandb.log(dict(
                (f'train/{name}_histogram', wandb.Histogram(out[name].detach().cpu().numpy().flatten())
                 ) for name in self.config.histogram_names
            ), step=self.step)
            
            self.net.eval()
            with torch.no_grad():
                val_metrics = defaultdict(list)
                for chrs, rets in self.dloader.get_split(self.config.val_start, self.config.test_start):
                    chrs = torch.from_numpy(chrs).to(self.config.device, non_blocking=True)
                    rets = torch.from_numpy(rets).to(self.config.device, non_blocking=True).unsqueeze(-1)
                    
                    out = self.net.predict(chrs, rets)
                    for k, v in extract_scalars(out).items():
                        val_metrics[k].append(v)
                val_metrics = mean_dict(val_metrics)
                self.log(val_metrics, prefix='val', suffix='mean', incr_step=False)
                
                wandb.log(dict(
                (f'val/{name}_histogram', wandb.Histogram(out[name].detach().cpu().numpy().flatten())
                 ) for name in self.config.histogram_names
                ), step=self.step)
                
                test_metrics = defaultdict(list)
                for chrs, rets in self.dloader.get_split(self.config.test_start, self.config.test_end):
                    chrs = torch.from_numpy(chrs).to(self.config.device, non_blocking=True)
                    rets = torch.from_numpy(rets).to(self.config.device, non_blocking=True).unsqueeze(-1)
                    
                    out = self.net.predict(chrs, rets)
                    for k, v in extract_scalars(out).items():
                        test_metrics[k].append(v)
                test_metrics = mean_dict(test_metrics)
                self.log(test_metrics, prefix='test', suffix='mean', incr_step=False)

                wandb.log(dict(
                (f'test/{name}_histogram', wandb.Histogram(out[name].detach().cpu().numpy().flatten())
                 ) for name in self.config.histogram_names
                ), step=self.step)
            pbar.set_description(f"val_loss: {val_metrics['loss']:.4f}, test_loss: {test_metrics['loss']:.4f}, val_pr2: {val_metrics['pred_r2']:.4f}, test_pr2: {test_metrics['pred_r2']:.4f}")

        self.save(f"./ckpts/{wandb.run.name}")