import torch


def entropy_loss(model, x, lambda_l1=1., lambda_entropy=2.):
    reg = 0
    postacts = model.postactivations(x)
    for i, pa in enumerate(postacts):
        layer = model.layers[i]
        input_range = layer.basis.grid[..., -1] - layer.basis.grid[..., 0] + 1e-4
        phi_norm = torch.mean(torch.abs(pa), dim=0) / input_range
        phi_sum = torch.sum(phi_norm)
        entropy = - torch.sum(phi_norm * torch.log2(phi_norm + 1e-4))
        #coef = torch.sum(torch.mean(torch.abs(layer.C), dim=1))
        #smooth = torch.sum(torch.mean(torch.abs(torch.diff(layer.C)), dim=1))
        reg += lambda_l1 * phi_sum + lambda_entropy * entropy
    return reg