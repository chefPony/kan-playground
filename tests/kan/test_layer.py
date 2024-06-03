import torch
from kan.layers import KANLayer
from kan.spline import SplineBasis


def test_update_grid():
    x = torch.stack([torch.arange(0, 2, 0.1), torch.arange(2, 4, 0.1)]).T
    grid = torch.stack([torch.arange(0, 1., 0.25), torch.arange(2, 3., 0.25)])
    spline = SplineBasis(grid=grid, k=3)
    l = KANLayer(n_in=2, n_out=4, basis=spline)
    l.update_from_samples(x, 0.1)