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


@torch.no_grad()
def gaussian_nll(model, loader, device="cpu", n_samples=20, min_var=1e-6):
    """
    Average negative log-likelihood of the targets under a Normal(mean, var)
    fit from `n_samples` stochastic forward passes per input.
    Score predictive uncertainty, not just point accuracy.
    """
    model.eval()

    total_nll, n = 0.0, 0
    for x, y in loader:
        mean, var = predictive_moments(model, x, n_samples, device)
        mean, var = mean.cpu(), var.clamp(min=min_var).cpu()

        nll = 0.5 * (torch.log(2 * math.pi * var) + (y - mean) ** 2 / var)
        total_nll += nll.sum().item()
        n += y.numel()

    return total_nll / n
