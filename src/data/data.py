"""
██████  ██████  ██████  ███    ██
     ██ ██   ██      ██ ████   ██
 █████  ██████   █████  ██ ██  ██
██      ██   ██ ██      ██  ██ ██
███████ ██████  ███████ ██   ████

Just a bunch of useful simple and useful datasets
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from ucimlrepo import fetch_ucirepo

_UCI_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_uci_cache")

class Dataset:
    """Generic Dataset function"""
    def __init__(self,
                 n_samples,
                 train_prop,
                 val_prop,
                 batch_size,
                 input_dim,
                 shuffle,
                 *args,
                 **kwargs
                 ):
        self.n_samples = n_samples
        self.train_size = int(train_prop * n_samples)
        self.val_size = int(val_prop * n_samples)
        self.test_size = n_samples - self.train_size - self.val_size
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.shuffle = shuffle


    def _function(self, x):
        pass


    def generate_data(self, min : int = -4, max : int = 4):
        x = torch.rand(self.n_samples, self.input_dim)*(max-min) + min
        y = self._function(x)
        y = (y - y.mean()) / y.norm()        
        dataset = TensorDataset(x, y)

        train_data, val_data, test_data = random_split(dataset, [self.train_size, self.val_size, self.test_size])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=self.shuffle) if self.test_size > 0 else None

        return train_loader, val_loader, test_loader, train_data, val_data, test_data



class DiscreteNoisyDataset(Dataset):
    """
    Discrete noisy dataset
        y = x + noise
    """
    def __init__(self,
                 n_samples,
                 train_prop,
                 val_prop,
                 batch_size,
                 input_dim,
                 shuffle,
                 sigma_squared,
                 choices,
                 *args,
                 **kwargs):
        super().__init__(n_samples, train_prop, val_prop, batch_size, input_dim, shuffle, *args, **kwargs)
        self.sigma_squared = sigma_squared
        self.choices = choices

    def _function(self, x):
        noise = torch.randn(x.shape) * self.sigma_squared
        y = x + noise
        return y


    def generate_data(self):
        idx = torch.randint(0, len(self.choices), (self.n_samples, self.input_dim))
        x = torch.tensor(self.choices)[idx]
        y = self._function(x)

        y = torch.nn.functional.normalize(y, dim = 0)

        dataset = TensorDataset(x, y)

        train_data, val_data, test_data = random_split(dataset, [self.train_size, self.val_size, self.test_size])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=self.shuffle) if self.test_size > 0 else None

        return train_loader, val_loader, test_loader, train_data, val_data, test_data


class SinusoidData(Dataset):
    """
    Sum of sinusoid data
        sin(x)+ 0.5 sin(2x) + 0.25 x^2
    """
    def __init__(self,
                 n_samples,
                 train_prop,
                 val_prop,
                 batch_size,
                 input_dim,
                 shuffle,
                 noise : bool = False,
                 *args,
                 **kwargs):
        super().__init__(n_samples, train_prop, val_prop, batch_size, input_dim, shuffle, *args, **kwargs)
        self.noise_value = 0.1


    def _function(self, x):
        y = torch.sin(x) + 0.5 * torch.sin(2 * x) + 0.1 * x **2
        y += torch.randn(y.shape) * self.noise_value
        return y



class SinusoidGapData(Dataset):
    """
    Sum of sinusoid data with a gap in [LowerBound, UpperBound], meaning that all the samples are outside this range
        sin(x)+ 0.5 sin(2x) + 0.25 x^2
    """
    def __init__(self,
                 n_samples,
                 train_prop,
                 val_prop,
                 batch_size,
                 input_dim,
                 shuffle,
                 lower_bound,
                 upper_bound,
                 noise : bool = False,
                 *args,
                 **kwargs):

        super().__init__(n_samples, train_prop, val_prop, batch_size, input_dim, shuffle, *args, **kwargs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def _function(self, x):
        y = torch.sin(x) + 0.5 * torch.sin(2 * x) + 0.1 * x **2
        return y


    def generate_data(self, minimum : float = -4., maximum : float = 4.):
        x = (maximum - minimum) * torch.rand(size=(self.n_samples, self.input_dim)) + minimum

        y = self._function(x)
        y += torch.randn(y.shape) * 0.1

        y = torch.nn.functional.normalize(y, dim = 0)

        dataset = TensorDataset(x, y)

        train_data, val_data, test_data = random_split(dataset, [self.train_size, self.val_size, self.test_size])
        x_train = dataset.tensors[0][train_data.indices]
        y_train = dataset.tensors[1][train_data.indices]
        gap_mask = (x_train < self.lower_bound) | (x_train > self.upper_bound)

        x_train_filtered = x_train[gap_mask].view(-1, self.input_dim)
        y_dim = y.shape[1] if len(y.shape) > 1 else 1
        y_train_filtered = y_train[gap_mask].view(-1, y_dim)
        train_data_filtered = TensorDataset(x_train_filtered, y_train_filtered)

        train_loader = DataLoader(train_data_filtered, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=self.shuffle) if self.test_size > 0 else None

        return train_loader, val_loader, test_loader, train_data, val_data, test_data



class UCIRegressionDataset(Dataset):
    """
    Loader for the UCI regression benchmark suite
    """

    def __init__(self,
                 uci_id: int,
                 train_prop: float = 0.8,
                 val_prop: float = 0.1,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 seed: int = 0,
                 target_index: int = 0,
                 *args,
                 **kwargs):
        

        os.makedirs(_UCI_CACHE_DIR, exist_ok=True)
        cache = os.path.join(_UCI_CACHE_DIR, f"uci_{uci_id}.pkl")
        if os.path.exists(cache):
            features, targets = pd.read_pickle(cache)
        else:
            data = fetch_ucirepo(id=uci_id).data
            features, targets = data.features, data.targets
            pd.to_pickle((features, targets), cache)

        X = torch.tensor(features.to_numpy(dtype=np.float32))
        y = torch.tensor(targets.iloc[:, target_index].to_numpy(dtype=np.float32)).unsqueeze(1)

        kwargs.pop('n_samples', None)
        kwargs.pop('input_dim', None) # You also pass input_dim positionally as X.shape[1]!
        kwargs.pop('output_dim', None)

        super().__init__(len(X), train_prop, val_prop, batch_size, X.shape[1], shuffle, *args, **kwargs)
        self.output_dim = y.shape[1]

        perm = np.random.default_rng(seed).permutation(self.n_samples)
        train_idx = perm[:self.train_size]
        val_idx = perm[self.train_size:self.train_size + self.val_size]
        test_idx = perm[self.train_size + self.val_size:]

        self.x_mean = X[train_idx].mean(0, keepdim=True)
        self.x_std = X[train_idx].std(0, keepdim=True).clamp(min=1e-8)
        self.y_mean = y[train_idx].mean(0, keepdim=True)
        self.y_std = y[train_idx].std(0, keepdim=True).clamp(min=1e-8)

        X = (X - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std

        self.train_data = TensorDataset(X[train_idx], y[train_idx])
        self.val_data = TensorDataset(X[val_idx], y[val_idx])
        self.test_data = TensorDataset(X[test_idx], y[test_idx])

    def generate_data(self):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def _get_dims(self):
        input_dim = self.train_data.tensors[0].shape[1]
        output_dim = self.train_data.tensors[1].shape[1]
        return input_dim, output_dim


UCI_DATASETS = {
    "concrete": 165,       # Concrete Compressive Strength   1030 x 8
    "energy": 242,         # Energy Efficiency                768 x 8  (targets Y1, Y2)
    "power_plant": 294,    # Combined Cycle Power Plant       9568 x 4
    "wine_quality": 186,   # Wine Quality (red+white merged)  6497 x 11
    "airfoil": 291,        # Airfoil Self-Noise               1503 x 5
    "auto_mpg": 9,         # Auto MPG                          398 x 7
}

if __name__ == "__main__":
    for name, uci_id in UCI_DATASETS.items():
        ds = UCIRegressionDataset(uci_id, batch_size=64)
        xb, yb = next(iter(ds.generate_data()[0]))
        print(f"{name:12s} in={ds.input_dim} out={ds.output_dim} "
              f"train/val/test={len(ds.train_data)}/{len(ds.val_data)}/{len(ds.test_data)} "
              f"batch_x={tuple(xb.shape)}")
