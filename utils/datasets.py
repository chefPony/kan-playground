import torch
from torch.utils.data import Dataset
from typing import Callable, List


class FunctionDataset(Dataset):

    def __init__(self, n_samples: int, n_vars: int, function: Callable, x_range: List[float] = None):
        super().__init__()
        self.X = torch.rand(n_samples, n_vars)
        if x_range:
            lb, ub = x_range
            self.X = (ub - lb) * self.X + lb
        self.y = function(self.X).reshape((-1, 1))

    def __getitem__(self, idx):
        return self.X[idx, ...], self.y[idx]

    def __len__(self):
        return self.X.shape[0]