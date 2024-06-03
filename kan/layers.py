import torch
from kan.spline import SplineBasis


class KANLayer(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, basis: SplineBasis):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.basis = basis

        c = torch.empty((self.n_in * self.n_out, self.basis.num_functions))
        torch.nn.init.xavier_normal_(c)
        self.c = torch.nn.Parameter(c)

        w = torch.empty((n_in, n_out))
        torch.nn.init.xavier_normal_(w)
        self.w = torch.nn.Parameter(w)

    def initialize_from_samples(self, x, gamma, margin=0.1):
        self.basis.fit_grid(x, gamma, margin)

    def update_from_samples(self, x, gamma, margin=0.1):
        with torch.no_grad():
            x_pos, _ = x.sort(dim=0)
            bx = self.basis.evaluate(x_pos)
            y_eval = (bx.unsqueeze(-1) * self.c).sum(2)
            self.basis.fit_grid(x, gamma, margin)
            x_eval = torch.repeat_interleave(x_pos, dim=-1)
            c = self.basis.get_coef(x_eval, y_eval)

    def get_coef(self, x, y):
        """

        :param x: (num_samples, nin)
        :param y: (num_samples, nin*nout)
        :return: (n_basis, nin, nout)
        """
        bx = self.basis.evaluate(x)
        bx_eval = torch.einsum("ijk,q->ijqk", bx, torch.ones(self.n_out))
        x_eval = torch.einsum("ij,k->ijk", x, torch.ones(self.n_out))

    def forward(self, x):
        out = self.basis.evaluate(x)
        out = (out.unsqueeze(-1) * self.c).sum(2)
        out = self.w * (torch.nn.functional.silu(x)[..., None] + out)
        out = torch.sum(out, dim=1)
        return out
