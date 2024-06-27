import torch
from kan.spline import SplineBasis
from kan.layers import KANLayer


class KANNetwork(torch.nn.Module):

    def __init__(self, n, grid_points, grid_range, k, base=torch.nn.SiLU()):
        super(KANNetwork, self).__init__()
        self.layers = list()
        grid = torch.linspace(grid_range[0], grid_range[-1], grid_points).reshape(1, -1)
        for nin, nout in zip(n[:-1], n[1:]):
            basis = SplineBasis(k=k, grid=torch.repeat_interleave(grid, nin, dim=0))
            layer = KANLayer(nin, nout, basis, base=base, bias=True)
            self.layers.append(layer)
        self.layers = torch.nn.Sequential(*self.layers)

    @staticmethod
    def from_layers(layers):
        n = [l.n_in for l in layers] + [layers[-1].n_out]
        k = layers[0].basis.k
        net = KANNetwork(n, 10, (-1, 1), k)
        net.layers = layers
        return net

    def forward(self, x):
        for l in self.layers:
            x, _ = l(x)
        return x

    def postactivations(self, x):
        postacts = list()
        for l in self.layers:
            x, post_act = l(x)
            postacts.append(post_act)
        return postacts

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

    #def prune(self, incoming_scores, outgoing_scores, th=1e-2):
    #    new_dim = [self.layers[0].n_in]
    #    for iscore, oscore in zip(incoming_scores, outgoing_scores):
    #        prune = (iscore < th) | (oscore < th)
    #        new_dim.append(torch.sum(prune).item())
    #    new_dim.append(self.layers[-1].n_out)
    #    new_net = KANNetwork(new_dim, grid_points=10, grid_range=[-1, 1], k=0)
    #    new_layers = list()
    #    for l, old_layer in enumerate(self.layers):
    #        prune = (iscore < th) | (oscore < th)
    #        new_layer = KANLayer(old_layer.n_in, new_dim[l], basis=old_layer.basis, C=)
    #        new_net
    #    return new_dim



