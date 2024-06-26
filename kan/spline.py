import torch


def _extend_grid(grid, k):
    h = (grid[:, -1] - grid[:, 0])/(grid.shape[1] - 1)
    if k > 0:
        h = h.unsqueeze(-1) * torch.arange(1, k+1)
        lext = grid[:, [0]] - h.flip(dims=(1, ))
        rext = grid[:, [-1]] + h
        grid = torch.cat([lext, grid, rext], dim=1)
    return grid


class SplineBasis:

    def __init__(self, grid: torch.Tensor, k: int):
        """
        A spline basis class

        Args:
            grid: (num_splines, num_knots)
            k: spline order
            extend_grid: if the provided grid needs to be extended considering the order
        """
        self.grid = grid
        self.k = k
        self.num_functions = self.grid.shape[1] - self.k - 1
        self.n_knots = self.grid.shape[1]

    def _evaluate(self, x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
        """
        Evaluate the spline basis over x, for given order k

        :param x: (num_samples, n_splines)
        :param grid: (n_splines, n_grid_points)
        :param k: spline order
        :return: (num_samples, n_splines, n_basis) where n_basis is n_grid_points - 1
        """
        if k == 0:
            value = (x >= grid[..., :-1]) * (x < grid[..., 1:])
        else:
            value = self._evaluate(x, grid, k - 1)
            t0 = (x - grid[..., :-(k + 1)]) / (grid[..., k:-1] - grid[..., :-(k + 1)])
            t1 = (grid[..., k + 1:] - x) / (grid[..., k + 1:] - grid[..., 1:(-k)])
            value = t0 * value[..., :-1] + t1 * value[..., 1:]
        return value

    def evaluate(self, x: torch.Tensor, extend_grid: bool = True) -> torch.Tensor:
        """
        Evaluate the spline basis over x

        :param x: (num_samples, n_splines)
        :return: (num_samples, n_splines, n_basis)
        """
        if extend_grid:
            grid = _extend_grid(self.grid, self.k).unsqueeze(dim=0)
        else:
            grid = self.grid.unsqueeze(dim=0)
        x = x.unsqueeze(dim=-1)
        return self._evaluate(x, grid, self.k)

    def evaluate_coef(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the spline basis over x for the given set of coefficients

        :param x: (num_samples, num_splines)
        :param c: (n_basis, num_splines)
        :return: (num_samples, num_splines)
        """
        bx = self.evaluate(x)  # (num_samples, nin, n_basis)
        out = torch.einsum('kij, ji->ki', bx.float(), c)
        return out

    def get_coef(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Retrieve the set of spline coefficients that would get y from x

        :param x: (num_samples, num_splines)
        :param y: (num_samples, num_splines)
        :return: (n_basis, num_splines)
        """
        # (num_samples, n_splines, n_basis)
        bx = self.evaluate(x)
        # (n_splines, num_samples, n_basis), (n_splines, num_samples, 1)
        res = torch.linalg.lstsq(bx.float().permute(1, 0, 2), y.permute(1, 0))
        return res.solution.T

    def fit_grid(self, x: torch.Tensor, n_knots: int, gamma: float, margin: float = 0.1) -> None:
        """
        Update the grid from samples
        :param x: (num_samples, n_splines)
        :param n_knots:
        :param gamma:
        :param margin:
        :return:
        """
        n_samples, n_grid_points = x.shape[0], n_knots
        step = n_samples // (n_grid_points - 1)
        idx = [i*step for i in range(n_grid_points-1)]
        x_pos, _ = x.sort(dim=0)
        adaptive_grid = torch.cat([x_pos[idx, :], x_pos[[-1], :]], dim=0)
        adaptive_grid[0, :] -= margin
        adaptive_grid[-1, :] += margin
        regular_grid = (x_pos[-1, :] - x_pos[0, :] + 2 * margin) / (n_grid_points - 1)
        regular_grid = x_pos[0, :] - margin + regular_grid * torch.arange(0, n_grid_points).unsqueeze(dim=-1)
        new_grid = regular_grid * (1 - gamma) + gamma * adaptive_grid
        self.n_knots = n_knots
        self.grid = new_grid.T

    def duplicate(self, num):
        grid = torch.repeat_interleave(self.grid, num, dim=0)
        return SplineBasis(grid=grid, k=self.k)


def curve2coef(x: torch.Tensor, y: torch.Tensor, basis: SplineBasis) -> torch.Tensor:
    """

    :param x: (n_samples, n_splines)
    :param y: (n_samples, n_splines)
    :param basis:
    :return:
    """
    # (num_samples, n_splines, n_basis)
    bx = basis.evaluate(x)
    c = torch.linalg.lstsq(bx, y.T).T
    return c
