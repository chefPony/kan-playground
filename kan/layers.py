import torch
from kan.spline import SplineBasis


class KANLayer(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, basis: SplineBasis, base = torch.nn.SiLU(),
                 noise_scale: float = 0.1, bias: bool = False, C: torch.Tensor = None, w_base: torch.Tensor = None,
                 w_sp: torch.Tensor = None):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.basis = basis

        noises = (torch.rand(self.basis.n_knots, self.n_in * self.n_out) * 2 - 1) * noise_scale
        if C is None:
            grid = torch.repeat_interleave(self.basis.grid.T, repeats=self.n_out, dim=1)
            C = self.basis.duplicate(self.n_out).get_coef(grid, noises)
        self.C = torch.nn.Parameter(C.contiguous())

        if w_base is None:
            w_base = torch.empty((1, self.n_in * self.n_out))
            torch.nn.init.xavier_normal_(w_base)
        self.w_base = torch.nn.Parameter(w_base)

        if w_sp is None:
            w_sp = torch.empty((1, self.n_in * self.n_out))
            torch.nn.init.xavier_normal_(w_sp)
        self.w_sp = torch.nn.Parameter(w_sp)
        self.base = base

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, self.n_out))
        else:
            self.bias = torch.zeros(1, self.n_out, requires_grad=False)

    def initialize_from_samples(self, x, gamma, margin=0.1):
        self.basis.fit_grid(x, self.basis.n_knots, gamma, margin)

    def update_from_samples(self, x, gamma, margin=0.1):
        with torch.no_grad():
            x_pos, _ = x.sort(dim=0)
            y_eval = self.spline(x_pos)
            self.basis.fit_grid(x_pos, self.basis.n_knots, gamma, margin)
            basis_dupl = self.basis.duplicate(self.n_out)
            x_eval = torch.repeat_interleave(x_pos, self.n_out, 1)
            C = basis_dupl.get_coef(x_eval, y_eval)
            self.C = torch.nn.Parameter(C)

    def refine_grid(self, x, num_points):
        with torch.no_grad():
            old_grid = self.basis.grid
            y_eval = self.spline(x)
            n_splines = self.basis.grid.shape[0]

            grid = torch.linspace(-1, 1, old_grid.shape[1]).reshape((1, -1))
            mapping = SplineBasis(grid=torch.repeat_interleave(grid, repeats=n_splines, dim=0), k=1)
            c_mapping = mapping.get_coef(mapping.grid.T, old_grid.T)
            percentiles = torch.repeat_interleave(torch.linspace(-1, 1, num_points).reshape(1, -1), n_splines, dim=0)
            new_grid = mapping.evaluate_coef(percentiles.T, c_mapping).T

            new_basis = SplineBasis(grid=new_grid, k=self.basis.k)
            basis_dupl = new_basis.duplicate(self.n_out)
            x_eval = torch.repeat_interleave(x, self.n_out, 1)
            C = basis_dupl.get_coef(x_eval, y_eval)
            self.basis = new_basis
            self.C = torch.nn.Parameter(C)

    def forward(self, x):
        x = self.w_sp * self.spline(x) + self.w_base * self.base_activation(x)
        post_act = x.reshape((x.shape[0], self.n_in, self.n_out))
        x = post_act.sum(dim=1) + self.bias
        return x, post_act

    def spline(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (num_samples, n_in)

        Returns:
            (num_samples, n_in * n_out)
        """
        x = torch.repeat_interleave(x, self.n_out, dim=1)
        out = self.basis.duplicate(self.n_out).evaluate_coef(x, self.C)
        return out

    def base_activation(self, x):
        x_eval = torch.repeat_interleave(x, self.n_out, dim=1).flatten(1)
        return self.base(x_eval)
