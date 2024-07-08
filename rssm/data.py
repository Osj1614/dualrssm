from dataclasses import dataclass

from typing import List, Iterable
from functools import cache

import numpy as np
import pandas as pd
import torch
import tqdm



class StockSeqDataLoader:
    @dataclass
    class Config:
        data_path: str = 'raw_data/data_mynp.npz'
        batch_size: int = 16
        stock_size: int = 512
        seq_len: int = 12
        rank_return_same_time: bool = False

        def __post_init__(self):
            if self.rank_return_same_time:
                self.seq_len += 1


    def __init__(self, config: Config = Config()):
        self.config = config
        print("Loading data...")
        npz = np.load(config.data_path, allow_pickle=True)

        self.ranks = npz['ranks']
        self.chrs = npz['characteristics']
        self.returns = np.clip(np.nan_to_num(npz['returns'], 0.0), -0.5, 2.0)
        self.dates = npz['dates']
        self.permnos = npz['permnos']
        self.columns = npz['columns']
        self.valids = npz['valids']

        seq_valids = []
        for t, date in enumerate(self.dates[:-self.config.seq_len]):
            seq_valids.append(self.valids[t:t+self.config.seq_len].all(axis=0))

        self.seq_valids = np.array(seq_valids)
        self.valid_steps = np.argwhere(self.seq_valids.sum(axis=1) > self.config.stock_size).flatten()


    def __len__(self):
        return len(self.valid_steps)


    def __getitem__(self, idx: int):
        t = self.valid_steps[idx]
        mask = np.random.choice(np.argwhere(self.seq_valids[t]).flatten(), size=self.config.stock_size, replace=False)
        return (
            self.ranks[t:t + self.config.seq_len - int(self.config.rank_return_same_time), mask], 
            self.returns[t - int(self.config.rank_return_same_time):t + self.config.seq_len - 2*int(self.config.rank_return_same_time), mask]
        )


    def get_split(self, start_date: str = None, end_date: str = None):
        if start_date is None:
            start_date = self.dates[self.valid_steps[0]]
        if end_date is None:
            end_date = self.dates[self.valid_steps[-1]]
        
        indices = np.argwhere((self.dates[self.valid_steps] >= pd.to_datetime(start_date)) & (self.dates[self.valid_steps] <= pd.to_datetime(end_date))).flatten()
        np.random.shuffle(indices)
        
        for i in range((len(indices) // self.config.batch_size) + bool(len(indices) % self.config.batch_size)):
            chrs, rets = [], []
            
            for idx in indices[i * self.config.batch_size:(i + 1) * self.config.batch_size]:
                chr_, ret = self[idx]
                chrs.append(chr_)
                rets.append(ret)
            yield np.stack(chrs, axis=0), np.stack(rets, axis=0)


    def get_full(self, idx: int):
        t = self.valid_steps[idx]
        mask = self.seq_valids[t]
        return (
            self.ranks[t:t + self.config.seq_len - int(self.config.rank_return_same_time), mask], 
            self.returns[t - int(self.config.rank_return_same_time):t + self.config.seq_len - 2*int(self.config.rank_return_same_time), mask]
        )
    


if __name__ == "__main__":
    dloader = StockSeqDataLoader(StockSeqDataLoader.Config())
    for batch in tqdm.tqdm(dloader):
        pass