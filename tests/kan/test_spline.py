import pytest
import torch
from kan.spline import SplineBasis


class TestSplineBasis:
    def test_evaluate(self):
        x = torch.stack([torch.arange(0, 1, 0.1), torch.arange(2, 3, 0.1)]).T
        grid = torch.stack([torch.arange(0, 1.25, 0.25), torch.arange(2, 3.25, 0.25)])

        spline = SplineBasis(grid=grid, k=0)
        out = spline.evaluate(x)

        expected_output = torch.Tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                                        ]).T
        expected_output = torch.stack([expected_output, expected_output], dim=1)
        torch.testing.assert_close(out.float(), expected_output.float())

    def test_coef_evaluate(self):
        x = torch.stack([torch.arange(0, 1, 0.1), torch.arange(2, 3, 0.1)]).T
        grid = torch.stack([torch.arange(0, 1.25, 0.25), torch.arange(2, 3.25, 0.25)])
        c = torch.ones((4, 2)) * 0.5
        spline = SplineBasis(grid=grid, k=0)
        y = spline.evaluate_coef(x, c)
        expected_y = torch.ones(x.shape) * 0.5
        torch.testing.assert_close(y, expected_y)

    def test_extend_grid(self):
        grid = torch.stack([torch.arange(0, 1., 0.25), torch.arange(2, 3., 0.25)])
        spline = SplineBasis(grid=grid, k=3)
        expect_grid = torch.Tensor([
            [-0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1, 1.25, 1.5],
            [1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3, 3.25, 3.5]
        ])
        torch.testing.assert_close(spline.grid, expect_grid)

    def test_get_coef(self):
        x = torch.stack([torch.arange(0, 1, 0.1), torch.arange(2, 3, 0.1)]).T
        grid = torch.stack([torch.arange(0, 1.25, 0.25), torch.arange(2, 3.25, 0.25)])
        expected_c = torch.Tensor([[0.3, 0.1], [1.2, 1.], [-0.2, 0.5], [1.4, 2.]])
        spline = SplineBasis(grid=grid, k=0)
        y = spline.evaluate_coef(x, expected_c)
        c = spline.get_coef(x, y)
        torch.testing.assert_close(c, expected_c)

    def test_fit_adaptive_grid(self):
        x = torch.stack([torch.arange(0, 1, 0.1), torch.arange(2, 3, 0.1)]).T
        grid = torch.stack([torch.arange(2, 3.25, 0.25), torch.arange(0, 1.25, 0.25)])
        expected_grid = torch.Tensor([
            [-0.1, 0.2, 0.4, 0.6, 1.],
            [1.9, 2.2, 2.4, 2.6, 3]])
        spline = SplineBasis(grid=grid, k=0)
        spline.fit_grid(x, n_knots=spline.n_knots, gamma=1)
        torch.testing.assert_close(spline.grid, expected_grid)

    def test_fit_regular_grid(self):
        x = torch.stack([torch.linspace(0, 1., 10), torch.linspace(2, 3, 10)]).T
        grid = torch.stack([torch.linspace(2, 3, 5), torch.linspace(0, 1, 5)])
        expected_grid = torch.stack([
            torch.linspace(x[0, 0].item() - 0.1, x[-1, 0].item() + 0.1, 5),
            torch.linspace(x[0, 1].item() - 0.1, x[-1, 1].item() + 0.1, 5)],
            dim=0
        )
        spline = SplineBasis(grid=grid, k=0)
        spline.fit_grid(x, n_knots=spline.n_knots, gamma=0)
        torch.testing.assert_close(spline.grid, expected_grid)

    def test_duplicate(self):
        grid = torch.stack([torch.arange(2, 3.25, 0.25), torch.arange(0, 1.25, 0.25)])
        spline = SplineBasis(k=0, grid=grid)
        new_spline = spline.duplicate(2)
        expected_grid = torch.Tensor([
            [2, 2.25, 2.5, 2.75, 3],
            [2, 2.25, 2.5, 2.75, 3],
            [0, 0.25, 0.5, 0.75, 1],
            [0, 0.25, 0.5, 0.75, 1]
        ])
        torch.testing.assert_close(new_spline.grid, expected_grid)