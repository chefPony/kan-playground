import torch
from kan.layers import KANLayer
from kan.spline import SplineBasis


class TestKANLayer:

    def test_spline(self):
        spline = SplineBasis(k=0, grid=torch.stack([torch.linspace(0, 1.25, 4), torch.linspace(3, 4.25, 4)]))
        x = torch.stack([torch.linspace(0, 1, 5), torch.linspace(3, 4, 5)]).T
        C = torch.ones((spline.num_functions, 4))
        C[:, 2:] = 0.5
        kl = KANLayer(n_in=2, n_out=2, basis=spline, C=C)
        out = kl.spline(x)
        expected_out = torch.ones((5, 4))
        expected_out[:, 2:] = 0.5
        torch.testing.assert_close(out, expected_out)

    def test_silu(self):
        spline = SplineBasis(k=0, grid=torch.stack([torch.linspace(0, 1.25, 4), torch.linspace(3, 4.25, 4)]))
        x = torch.stack([torch.linspace(0, 1, 5), torch.linspace(3, 4, 5)]).T
        W = torch.ones((spline.num_functions, 4))
        kl = KANLayer(n_in=2, n_out=2, basis=spline, W=W)
        out = kl.silu(x)
        assert out.shape == (5, 4)

    def test_update_grid(self):
        x = torch.stack([torch.arange(0, 2, 0.1), torch.arange(2, 4, 0.1)]).T
        grid = torch.stack([torch.arange(0, 1., 0.25), torch.arange(2, 3., 0.25)])
        spline = SplineBasis(grid=grid, k=1)
        old_C = torch.ones((spline.num_functions, 4))
        l = KANLayer(n_in=2, n_out=2, basis=spline, C=old_C)
        y1 = l.spline(x)
        l.update_from_samples(x, gamma=1, margin=0.1)
        new_C = l.C
        y2 = l.spline(x)
        expected_grid = torch.Tensor([[-0.1, 0.4, 0.8, 1.2, 1.6, 2],
                                      [1.9, 2.4, 2.8, 3.2, 3.6, 4]])
        torch.testing.assert_close(l.basis.grid, expected_grid)

