from dataclasses import dataclass, field
from collections import defaultdict
from functools import cached_property
from copy import deepcopy
import random

import torch
import torch.nn as nn
import numpy as np
import wandb
from tensorboardX import SummaryWriter
from info_nce import InfoNCE

from ae_pricing.typing import *
from ae_pricing.net import GKXModel, MeanEnsemble
from ae_pricing.data import GKXDataLoader
from ae_pricing.utils import *
from ae_pricing.eval import pred_r2_score
from ae_pricing.pcgrad import PCGrad



class AEPricingTrainer:
    @dataclass
    class Config:
        project_name: str = "ae_pricing5"
        wandb_mode: str = "online"
        exp_name: str = None

        data_dir: str = "./raw_data/data_mynew_shifted.npz"
        device: str = "cuda:0"

        train_timesteps: int = 12 * 13
        val_timesteps: int = 12 * 2
        test_timesteps: int = 12 * 1
        incr_timesteps: int = 12 * 1  # Better match with test_timesteps
        extend_trainset: bool = True

        n_epochs: int = 1000
        batch_size: int = 512
        early_stop_patience: int = 15  # Epochs
        best_criteria: str = "loss"  # "loss", "total_r2", "pred_r2"

        lr: float = 1e-3
        l1_reg: float = 1e-6
        n_ensemble: int = 8
        n_factors: int = 1
        nce_loss: bool = False
        nce_loss_weight: float = 0.001
        do_pcgrad: bool = True

        beta_arch: list = field(default_factory=lambda: [32, 16, 8])
        factor_arch: list = field(default_factory=lambda: [])  # single layer
        seed: int = None
        reproducible: bool = False 

        @cached_property
        def best_sign(self) -> float:
            if "r2" in self.best_criteria:
                return -1.0
            elif self.best_criteria == "loss":
                return 1.0
            else:
                raise ValueError(f"Unknown best criteria: {self.best_criteria}")


    model: MeanEnsemble
    optim: torch.optim.Optimizer

    def __init__(self, cfg: Config=Config()):
        if cfg.seed is None:
            cfg.seed = np.random.randint(0, 100000000)

        if cfg.reproducible:  # may slow down training
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
            np.random.seed(cfg.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            random.seed(cfg.seed)

        self.cfg = cfg
        self.dloader = GKXDataLoader(cfg.data_dir, cfg.batch_size, seed=cfg.seed)
        self.num_chrs = self.dloader[0][0].shape[1]
        self.loss_fn = nn.MSELoss()
        self.infonce = InfoNCE(negative_mode='unpaired')
    

    def init_model(self):
        self.model = MeanEnsemble([
            GKXModel(
                beta_arch=self.cfg.beta_arch, factor_arch=self.cfg.factor_arch, num_chrs=self.num_chrs, num_factors=self.cfg.n_factors
            )  for _ in range(self.cfg.n_ensemble)
        ]).to(self.cfg.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        if self.cfg.do_pcgrad:
            self.optim = PCGrad(self.optim)


    def train_epoch(self, train_dloader) -> InfoDict:
        info = defaultdict(float)
        self.model.train()

        for chrs, rets in train_dloader:
            chrs, rets = tuple_arr_to_tensor((chrs, rets), dtype=torch.float32, device=self.cfg.device)
            self.optim.zero_grad()
            (pred_rets, betas, factors, port_rets, _), _ = self.model(chrs, rets)
            loss = self.loss_fn(rets, pred_rets)
            loss.backward()
            self.optim.step()

            info["loss"] += loss.item()
            with torch.no_grad():
                total_r2 =  1 - (pred_rets - rets).square().sum().item() / rets.square().sum().item()
                info["total_r2"] += total_r2 * 100

        for k in info:
            info[k] /= len(train_dloader)
        return info


    def train_epoch_contrastive(self, train_dloader) -> InfoDict:
        info = defaultdict(float)
        self.model.train()

        for (chrs, rets), (chrs2, rets2) in zip(train_dloader, train_dloader):
            chrs, rets = tuple_arr_to_tensor((chrs, rets), dtype=torch.float32, device=self.cfg.device)
            chrs2, rets2 = tuple_arr_to_tensor((chrs2, rets2), dtype=torch.float32, device=self.cfg.device)

            chrs_0, chrs_1 = torch.split(chrs, chrs.shape[0] // 2, dim=0)
            rets_0, rets_1 = torch.split(rets, rets.shape[0] // 2, dim=0)
            chrs2_0, chrs2_1 = torch.split(chrs2, chrs2.shape[0] // 2, dim=0)
            rets2_0, rets2_1 = torch.split(rets2, rets2.shape[0] // 2, dim=0)
            
            self.optim.zero_grad()
            (pred_rets_0, _, factors_0, _, _), (_, _, _, _, factor_inters_0) = self.model(chrs_0, rets_0)
            (pred_rets_1, _, factors_1, _, _), (_, _, _, _, factor_inters_1)= self.model(chrs_1, rets_1)
            (pred_rets2_0, _, factors2_0, _, _), (_, _, _, _, factor_inters2_0) = self.model(chrs2_0, rets2_0)
            (pred_rets2_1, _, factors2_1, _, _), (_, _, _, _, factor_inters2_1) = self.model(chrs2_1, rets2_1)

            factors_0, factors_1, factors2_0, factors2_1 = factors_0.unsqueeze(0), factors_1.unsqueeze(0), factors2_0.unsqueeze(0), factors2_1.unsqueeze(0)

            pred_loss = self.loss_fn(rets_0, pred_rets_0)
            pred_loss += self.loss_fn(rets_1, pred_rets_1)
            pred_loss += self.loss_fn(rets2_0, pred_rets2_0)
            pred_loss += self.loss_fn(rets2_1, pred_rets2_1)

            f_inters = torch.cat([factor_inters_0, factor_inters_1], dim=0)
            f_inters2 = torch.cat([factor_inters2_0, factor_inters2_1], dim=0)
            nce_loss = self.infonce(f_inters, positive_key=f_inters, negative_keys=f_inters2) 
            nce_loss += self.infonce(f_inters2, positive_key=f_inters2, negative_keys=f_inters)

            if self.cfg.do_pcgrad:
                losses = [pred_loss, self.cfg.nce_loss_weight * nce_loss]
                loss = pred_loss + self.cfg.nce_loss_weight * nce_loss
                self.optim.pc_backward(losses)
            else:
                loss = pred_loss + self.cfg.nce_loss_weight * nce_loss
                loss.backward()
            self.optim.step()

            info["loss"] += loss.item()
            info["pred_loss"] += pred_loss.item()
            info["nce_loss"] += self.cfg.nce_loss_weight * nce_loss.item()

            with torch.no_grad():
                pred_rets = torch.cat([pred_rets_0, pred_rets_1, pred_rets2_0, pred_rets2_1], dim=0)
                rets = torch.cat([rets_0, rets_1, rets2_0, rets2_1], dim=0)
                total_r2 =  1 - (pred_rets - rets).square().sum().item() / rets.square().sum().item()
                info["total_r2"] += total_r2 * 100

        for k in info:
            info[k] /= len(train_dloader)
        return info
    

    def eval_epoch(self, val_dloader) -> InfoDict:
        info = defaultdict(float)
        self.model.eval()

        with torch.no_grad():
            for chrs, rets in val_dloader:
                chrs, rets = tuple_arr_to_tensor((chrs, rets), dtype=torch.float32, device=self.cfg.device)
                (pred_rets, betas, factors, port_rets, _), _ = self.model(chrs, rets)
                info["loss"] += self.loss_fn(rets, pred_rets).item()
                info["total_r2"] += (1 - (pred_rets - rets).square().sum().item() / rets.square().sum().item()) * 100

        for k in info:
            info[k] /= len(val_dloader)
        return info


    def run_exp_on_split(self, train_start, val_start, val_end, test_end):
        _, train_dloader, val_dloader, test_dloader, _ = self.dloader.split([train_start, val_start, val_end, test_end])
        train_dloader.use_full_batch = False
        val_dloader.use_full_batch = True
        test_dloader.use_full_batch = True

        wandb.define_metric(f"run_{test_end}/step")
        # set all other train/ metrics to use this step
        wandb.define_metric(f"run_{test_end}/*", f"run_{test_end}/step")

        self.init_model()
        best_val_metric = self.cfg.best_sign * np.inf
        best_val_epoch = 0
        best_test_info = None
        best_model = None

        for epoch in range(self.cfg.n_epochs):
            if self.cfg.nce_loss:
                train_info = self.train_epoch_contrastive(train_dloader)
            else:
                train_info = self.train_epoch(train_dloader)
            val_info = self.eval_epoch(val_dloader)
            test_info = self.eval_epoch(test_dloader)
            
            val_info["pred_r2"] = pred_r2_score(self.model, self.dloader, train_dloader.dates, val_dloader.dates, device=self.cfg.device) * 100
            test_info["pred_r2"] = pred_r2_score(self.model, self.dloader, train_dloader.dates, test_dloader.dates, device=self.cfg.device) * 100
            
            if val_info[self.cfg.best_criteria] * self.cfg.best_sign < best_val_metric * self.cfg.best_sign:
                best_val_metric = val_info[self.cfg.best_criteria]
                best_val_epoch = epoch
                best_test_info = test_info
                if best_model is not None: del best_model
                print(f"{test_end}: New best {self.cfg.best_criteria}: {best_val_metric:6.4f}, testset: {test_info[self.cfg.best_criteria]:6.4f}")
                best_model = deepcopy(self.model.state_dict())
            
            wandb.log({f"run_{test_end}/step": epoch})
            wandb.log(affix_dict(train_info, f"run_{test_end}/train_"))
            wandb.log(affix_dict(val_info, f"run_{test_end}/val_"))
            wandb.log(affix_dict(test_info, f"run_{test_end}/test_"), commit=True)

            if epoch - best_val_epoch > self.cfg.early_stop_patience:
                break
        
        wandb.log({f"eval/step": test_end})
        wandb.log(affix_dict(best_test_info, f"eval/"), commit=True)
        torch.save(best_model, f"{wandb.run.dir}/model_best_{test_end}.pt")

    
    def split_gen(self):
        dates = sorted(self.dloader.dates)
        
        train_start = 10 * 12
        val_start = train_start + self.cfg.train_timesteps
        val_end = train_start + self.cfg.train_timesteps + self.cfg.val_timesteps
        test_end = train_start + self.cfg.train_timesteps + self.cfg.val_timesteps + self.cfg.test_timesteps

        while test_end < len(dates):
            yield dates[train_start], dates[val_start], dates[val_end], dates[test_end]
            if not self.cfg.extend_trainset:
                train_start += self.cfg.incr_timesteps
            val_start += self.cfg.incr_timesteps
            val_end += self.cfg.incr_timesteps
            test_end += self.cfg.incr_timesteps
    

    def run(self):
        wandb.init(config=self.cfg, project=self.cfg.project_name, mode=self.cfg.wandb_mode, name=self.cfg.exp_name, 
                   tags=[f"{self.cfg.n_ensemble}_ensemble", f"{self.cfg.n_factors}_factors", f"stop_criteria__{self.cfg.best_criteria}"])
        #wandb.watch(self.model)
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")

        for train_start, val_start, val_end, test_end in self.split_gen():
            self.run_exp_on_split(train_start, val_start, val_end, test_end)
