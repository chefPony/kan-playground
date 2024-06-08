import torch
from kan.layers import KANLayer
from kan.spline import SplineBasis


class TestKANLayer:

    def test_spline(self):
        spline = SplineBasis(k=0, grid=torch.stack([torch.linspace(0, 1.25, 4), torch.linspace(3, 4.25, 4)]))
        x = torch.stack([torch.linspace(0, 1, 5), torch.linspace(3, 4, 5)]).T
        C = torch.ones((spline.num_functions, 4))
        C[:, 2:] = 0.5
        kl = KANLayer(n_in=2, n_out=2, basis=spline, C=C, w_sp=None)
        out = kl.spline(x)
        expected_out = torch.ones((5, 4))
        expected_out[:, 2:] = 0.5
        torch.testing.assert_close(out, expected_out)

    def test_silu(self):
        spline = SplineBasis(k=0, grid=torch.stack([torch.linspace(0, 1.25, 4), torch.linspace(3, 4.25, 4)]))
        x = torch.stack([torch.linspace(0, 1, 5), torch.linspace(3, 4, 5)]).T
        kl = KANLayer(n_in=2, n_out=2, basis=spline, w_sp=None)
        out = kl.silu(x)
        assert out.shape == (5, 4)

    def test_forward(self):
        spline = SplineBasis(k=0, grid=torch.stack([torch.linspace(0, 1.25, 4), torch.linspace(3, 4.25, 4)]))
        x = torch.stack([torch.linspace(0, 1, 5), torch.linspace(0, 1, 5)]).T
        C = torch.ones((spline.num_functions, 4))
        W = torch.ones((1, 4))
        layer = KANLayer(n_in=2, n_out=2, basis=spline, C=C, w_silu=W, w_sp=W)
        out = layer(x)
        s = torch.ones((5, 4))
        s[:, 2:] = 0
        s = s + layer.silu(x)
        expected_out = torch.zeros((5, 2))
        expected_out[:, 0] = s[:, 0] + s[:, 2]
        expected_out[:, 1] = s[:, 1] + s[:, 3]
        torch.testing.assert_close(out, expected_out)

    def test_refine_grid(self):
        grid = torch.linspace(-1, 1, 12)
        grid = torch.stack([grid, grid], dim=0)
        x = torch.linspace(-1, 1, 1000)
        x = torch.stack([x, x], dim=1)
        spline = SplineBasis(grid=grid, k=3, extend_grid=False)
        klay = KANLayer(2, 2, basis=spline)
        y1 = klay.spline(x)
        klay.refine_grid(x, 50)
        y2 = klay.spline(x)
        torch.testing.assert_close(y1, y2, atol=1e-3, rtol=1e-2)

