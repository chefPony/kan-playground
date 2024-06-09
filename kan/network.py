import torch
from kan.spline import SplineBasis
from kan.layers import KANLayer

class KANNetwork(torch.nn.Module):

    def __init__(self, n, grid, k):
        super(KANNetwork, self).__init__()
        self.layers = list()
        for nin, nout in zip(n[:-1], n[1:]):
            basis = SplineBasis(k=k, grid=torch.repeat_interleave(grid, nin, dim=0))
            layer = KANLayer(nin, nout, basis,  bias=True)
            self.layers.append(layer)
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        for l in self.layers:
            x, _ = l(x)
        return x

    def get_postactivation(self, x, layer: int):
        with torch.no_grad():
            for i in range(layer):
                x, post_act = self.layers[i](x)
            return post_act

    def initialize_from_samples(self, x, gamma, margin):
        with torch.no_grad():
            for l in self.layers:
                l.initialize_from_samples(x, gamma, margin)
                x, _ = l(x)

    def update_grid(self, x, gamma, margin):
        with torch.no_grad():
            for l in self.layers:
                l.update_from_samples(x, gamma, margin)
                x, _ = l(x)

    def refine_grid(self, x, num_points):
        with torch.no_grad():
            for l in self.layers:
                l.refine_grid(x, num_points)
                x, _ = l(x)