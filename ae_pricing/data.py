from typing import List, Iterable
from functools import cache

import numpy as np
import pandas as pd
import torch
import tqdm

from ae_pricing.typing import DateInt


class PandasDataLoader:
    def __init__(self, df: pd.DataFrame, append_date: bool=True, append_permno: bool=False):
        self.df = df
        #self.df = df.reset_index()
        
        self.dates = df.index.get_level_values("DATE").unique().sort_values()
        self.both_permnos = []
        self.data = []

        col_without_ret = df.columns.drop("RET")

        print("preparing data...")
        for date, next_date in tqdm.tqdm(zip(self.dates[:-1], self.dates[1:])):
            today_df = self.df.loc[(slice(None), date), :]
            next_df = self.df.loc[(slice(None), next_date), :]

            both_permnos = today_df.index.get_level_values("permno").intersection(next_df.index.get_level_values("permno"))
            self.both_permnos.append(both_permnos)

            chrs = today_df.loc[(both_permnos, slice(None)), col_without_ret].values  # (Assets, Characteristics)
            rets: np.ndarray = next_df.loc[(both_permnos, slice(None)), 'RET'].values[:, None]  # (Assets, 1)
            
            d = []
            if append_date:
                d.append(date)
            if append_permno:
                d.append(both_permnos)
            
            self.data.append((
                *d, np.concatenate((chrs, rets.mean(keepdims=True).repeat(chrs.shape[0], axis=0)), axis=1), rets
            ))


    def __len__(self):
        return len(self.dates) - 1
    

    @property
    def num_characteristics(self) -> int:
        return self.data[0][1].shape[1]


    def __getitem__(self, i: int):
        return self.data[i]


    
def batch_gen(pdloader: PandasDataLoader, indices=List[int], device="cpu"):
    while True:
        indices = np.random.permutation(indices)
        for i in indices:
            date, chrs, rets = pdloader[i]
            chrs, rets = torch.from_numpy(chrs).float().to(device, non_blocking=True), torch.from_numpy(rets).float().to(device, non_blocking=True)
            yield date, chrs, rets



class GKXDataLoader:
    def __init__(self, data_dir="", use_full_batch=False, batch_size=512,
                 shuffle=True, seed=42, load_data=True, add_noise=True) -> None:
        self.seed = seed
        self.data_dir = data_dir
        self.use_full_batch = use_full_batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.add_noise = add_noise

        self.prng = np.random.default_rng(seed)

        self.chrs: dict[int, np.ndarray] = {}  # {date: (Assets, Characteristics)}. date in format YYYYMMDD
        self.rets: dict[int, np.ndarray] = {}  # {date: (Assets, 1)}. date in format YYYYMMDD
        self.dates: list[int] = []  # [date]. date in format YYYYMMDD

        if load_data:
            self._load_data()


    def __len__(self):
        return len(self.dates)


    def __getitem__(self, i: int):
        return self.chrs[self.dates[i]], self.rets[self.dates[i]]


    def __iter__(self):
        indices = np.arange(len(self))

        if self.shuffle:
            indices = self.prng.permutation(indices)

        for i in indices:
            chrs, rets = self[i]

            if self.use_full_batch:
                yield chrs, rets
            else:
                # all number of assets must be greater than batch_size
                indices = self.prng.choice(chrs.shape[0], self.batch_size, replace=False)
                yield chrs[indices], rets[indices]


    def split(self, split_dates: Iterable[int]) -> List["GKXDataLoader"]:
        splits = [
            GKXDataLoader(
                self.data_dir, self.use_full_batch, self.batch_size, 
                self.shuffle, self.prng.integers(0, 10000000), False
            ) for _ in range(len(split_dates) + 1)
        ]

        for date in self.dates:
            for j, split_date in enumerate(split_dates):
                if date <= split_date:
                    splits[j].chrs[date] = self.chrs[date]
                    splits[j].rets[date] = self.rets[date]
                    splits[j].dates.append(date)
                    break
            else:
                splits[-1].chrs[date] = self.chrs[date]
                splits[-1].rets[date] = self.rets[date]
                splits[-1].dates.append(date)

        for split in splits:
            split.dates = np.array(split.dates)
        
        return splits


    def slice(self, start: DateInt, end: DateInt) -> "GKXDataLoader":
        new_loader = GKXDataLoader(self.data_dir, self.use_full_batch, self.batch_size, 
                                   self.shuffle, self.prng.integers(0, 10000000), False)
        new_loader.chrs = {date: self.chrs[date] for date in self.dates if start <= date <= end}
        new_loader.rets = {date: self.rets[date] for date in self.dates if start <= date <= end}
        new_loader.dates = np.array([date for date in self.dates if start <= date <= end])

        return new_loader
    

    @cache
    def as_whole(self):
        whole_chrs = np.concatenate([self.chrs[date] for date in self.dates])
        whole_rets = np.concatenate([self.chrs[date] for date in self.dates])
        whole_chrs.flags.writeable = False
        whole_rets.flags.writeable = False
        
        return whole_chrs, whole_rets


    def save(self, path: str):
        np.savez(path, chrs=self.chrs, rets=self.rets, dates=self.dates)


    def _load_data(self):
        print("Loading data...")
        if self.data_dir.split('.')[-1] == 'sas7bdat':
            self.data = pd.read_sas(self.data_dir)
        elif self.data_dir.split('.')[-1] == 'parquet':
            self.data = pd.read_parquet(self.data_dir)
        elif self.data_dir.split('.')[-1] == 'npz':
            self.data = np.load(self.data_dir, allow_pickle=True)
            self.chrs = self.data['chrs'].item()
            if self.add_noise:
                for date in self.chrs:
                    self.chrs[date] += self.prng.normal(0, 1e-5, self.chrs[date].shape)
            self.rets = self.data['rets'].item()
            self.dates = self.data['dates']
        else:
            raise NotImplementedError(f"Unknown file type: {self.data_dir.split('.')[-1]}")

        if type(self.data) == pd.DataFrame:
            self._parse_dataframe()
    

    def _parse_dataframe(self):
        assert type(self.data) == pd.DataFrame, "Data is not a pandas DataFrame"
        
        print("Parsing data...")
        for date in sorted(self.data.DATE.unique()):
            print(f"\r{date}", end="", flush=True)
            date_ = int(date.strftime("%Y%m%d"))
            self.chrs[date_] = self.data.loc[self.data.DATE == date, self.data.columns.drop(["RET", "permno", 'DATE'])].values.astype(np.float32)
            if self.add_noise:
                self.chrs[date_] += self.prng.normal(0, 1e-5, self.chrs[date_].shape)
            self.rets[date_] = self.data.loc[self.data.DATE == date, "RET"].values[:, None].astype(np.float32)
            self.dates.append(date_)
        self.dates = np.array(self.dates)
        print("\nDone.")