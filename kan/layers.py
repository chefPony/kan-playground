import torch
from kan.spline import SplineBasis


class KANLayer(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, basis: SplineBasis, C: torch.Tensor = None, W: torch.Tensor = None):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.basis = basis

        if C is None:
            C = torch.empty((self.basis.num_functions, self.n_in * self.n_out, ))
            torch.nn.init.xavier_normal_(C)
        self.C = torch.nn.Parameter(C)

        if W is None:
            W = torch.empty((1, self.n_in * self.n_out))
            torch.nn.init.xavier_normal_(W)
        self.W = torch.nn.Parameter(W)

    def initialize_from_samples(self, x, gamma, margin=0.1):
        self.basis.fit_grid(x, gamma, margin)

    def update_from_samples(self, x, gamma, margin=0.1):
        with torch.no_grad():
            x_pos, _ = x.sort(dim=0)
            y_eval = self.spline(x_pos)
            self.basis.fit_grid(x_pos, gamma, margin)
            basis_dupl = self.basis.duplicate(self.n_out)
            C = basis_dupl.get_coef(torch.repeat_interleave(x_pos, self.n_out, 1), y_eval)
            self.C = torch.nn.Parameter(C)

    def forward(self, x):
        out = self.W * (self.silu(x) + self.spline(x))
        return out

    def spline(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (num_samples, n_in)

        Returns:
            (num_samples, n_in * n_out)
        """
        ones = torch.ones(self.n_out)
        # (num_samples, nin, nbasis)
        out = self.basis.evaluate(x)
        # (n_samples, nin * nout, nbasis)
        out = torch.einsum("ijk,q->ijqk", out, ones).flatten(1, 2)
        # (n_samples, nin * nout)
        out = torch.einsum("ijk,kj->ij", out, self.C)
        return out

    def silu(self, x):
        ones = torch.ones(self.n_out)
        x_eval = torch.einsum("ij,k->ijk", x, ones).flatten(1)
        return torch.nn.functional.silu(x_eval)
