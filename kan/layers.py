import torch
from kan.spline import SplineBasis


class KANLayer(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, basis: SplineBasis, noise_scale: float = 0.1,
                 C: torch.Tensor = None, w_silu: torch.Tensor = None, w_sp: torch.Tensor = None):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.basis = basis

        noises = (torch.rand(self.basis.n_knots, self.n_in * self.n_out) * 2 - 1) * noise_scale
        if C is None:
            grid = torch.repeat_interleave(self.basis.grid.T, repeats=self.n_out, dim=1)
            C = self.basis.duplicate(self.n_out).get_coef(grid, noises)
        self.C = torch.nn.Parameter(C)

        if w_silu is None:
            w_silu = torch.empty((1, self.n_in * self.n_out))
            torch.nn.init.xavier_normal_(w_silu)
        self.w_silu = torch.nn.Parameter(w_silu)

        if w_sp is None:
            w_sp = torch.empty((1, self.n_in * self.n_out))
            torch.nn.init.xavier_normal_(w_sp)
        self.w_sp = torch.nn.Parameter(w_sp)

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
            c_mapping = mapping.get_coef(mapping.grid.T[1:-1, ...], old_grid.T)
            percentiles = torch.repeat_interleave(torch.linspace(-1, 1, num_points).reshape(1, -1), n_splines, dim=0)
            new_grid = mapping.evaluate_coef(percentiles.T, c_mapping).T

            new_basis = SplineBasis(grid=new_grid, k=self.basis.k, extend_grid=False)
            basis_dupl = new_basis.duplicate(self.n_out)
            x_eval = torch.repeat_interleave(x, self.n_out, 1)
            C = basis_dupl.get_coef(x_eval, y_eval)
            self.basis = new_basis
            self.C = torch.nn.Parameter(C)

    def forward(self, x):
        out = self.w_silu * self.silu(x) + self.w_sp * self.spline(x)
        out = out.reshape((x.shape[0], self.n_in, self.n_out))
        return out.sum(dim=1)

    def spline(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (num_samples, n_in)

        Returns:
            (num_samples, n_in * n_out)
        """
        # (num_samples, nin, nbasis)
        out = self.basis.evaluate(x)
        # (n_samples, nin * nout, nbasis)
        #out = torch.einsum("ijk,q->ijqk", out, ones).flatten(1, 2)
        out = torch.repeat_interleave(out, self.n_out, dim=1)
        # (n_samples, nin * nout)
        out = torch.einsum("ijk,kj->ij", out, self.C)
        return out

    def silu(self, x):
        ones = torch.ones(self.n_out)
        x_eval = torch.einsum("ij,k->ijk", x, ones).flatten(1)
        return torch.nn.functional.silu(x_eval)
