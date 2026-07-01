"""
██████  ██████  ██████  ███    ██
     ██ ██   ██      ██ ████   ██
 █████  ██████   █████  ██ ██  ██
██      ██   ██ ██      ██  ██ ██
███████ ██████  ███████ ██   ████

Just a bunch of useful simple and useful datasets
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class Dataset:
    """Generic Dataset function"""
    def __init__(self,
                 n_samples,
                 train_prop,
                 val_prop,
                 batch_size,
                 input_dim,
                 shuffle
                 ):
        self.n_samples = n_samples
        self.train_size = train_prop * n_samples
        self.val_size = val_prop * n_samples
        self.test_size = (1.-train_prop-val_prop) * n_samples
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.shuffle = shuffle


    def _function(self, x):
        pass


    def generate_data(self):
        x = torch.randn(self.n_samples, self.input_dim)
        y = self._function(x)

        dataset = TensorDataset(x, y)

        train_data, val_data = random_split(dataset, [self.train_size, self.val_size, self.test_size])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_loader, val_loader, train_data, val_data



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
                 shuffle):
        super().__init__(n_samples, train_prop, val_prop, batch_size, input_dim, shuffle)


    def _function(self, x):
        y = torch.sin(x) + 0.5 * torch.sin(2 * x) + 0.25 * x **2




