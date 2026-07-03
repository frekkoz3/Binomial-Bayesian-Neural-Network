r"""
██████  ██████  ██████  ███    ██
     ██ ██   ██      ██ ████   ██
 █████  ██████   █████  ██ ██  ██
██      ██   ██ ██      ██  ██ ██
███████ ██████  ███████ ██   ████
"""

import math
import torch

@torch.no_grad()
def predictive_moments(model, x, n_samples, device):
    """Mean and variance of `n_samples` repeated forward passes on `x`."""
    x = x.to(device)
    preds = torch.stack([model(x) for _ in range(n_samples)], dim=0)
    return preds.mean(dim=0), preds.var(dim=0, unbiased=False)


@torch.no_grad()
def rmse(model, loader, device="cpu", n_samples=1):
    """Root mean squared error of the mean against the target."""
    model.eval()

    squared_error, n = 0.0, 0
    for x, y in loader:
        mean, _ = predictive_moments(model, x, n_samples, device)
        squared_error += ((mean.cpu() - y) ** 2).sum().item()
        n += y.numel()

    return math.sqrt(squared_error / n)
