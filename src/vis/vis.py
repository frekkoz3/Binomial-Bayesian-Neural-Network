r"""
██████  ██████  ██████  ███    ██ 
     ██ ██   ██      ██ ████   ██ 
 █████  ██████   █████  ██ ██  ██ 
██      ██   ██ ██      ██  ██ ██ 
███████ ██████  ███████ ██   ████                                 
"""
import torch
import math
from matplotlib import pyplot as plt
from torch.distributions import Normal, Binomial


from src.val.metrics import *

def plot_history(history: dict, model : str, save : bool = False, path : str = None):
    """
    Plot the training and validation losses.

    Parameters
    ----------
    history : dict
        Dictionary returned by `fit()` containing:
            - "train_loss"
            - "val_loss"
    """

    plt.figure(figsize=(8, 5))

    plt.plot(history["train_loss"], label="Train", color="#2e7d4f")
    plt.plot(history["val_loss"], label="Validation", color="#8bc34a")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training History for {model}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=600)
    plt.show()


@torch.no_grad()
def plot_results(model, dataloader, device="cpu", save=False, path=None):
    """
    Plot predictions for a 1D regression problem with predictive uncertainty.

    Assumes:
        input_dim = 1
        output_dim = 1
    """

    model.eval()

    xs = []
    ys = []
    preds = []
    vars_ = []

    for x, y in dataloader:
        x = x.to(device)

        pred, var = predictive_moments(
            model,
            x,
            n_samples=100,
            device=device,
        )

        xs.append(x.cpu())
        ys.append(y.cpu())
        preds.append(pred.cpu())
        vars_.append(var.cpu())

    xs = torch.cat(xs).squeeze()
    ys = torch.cat(ys).squeeze()
    preds = torch.cat(preds).squeeze()
    vars_ = torch.cat(vars_).squeeze()

    if xs.ndim > 1:
        xs = xs[:, 0]
    if ys.ndim > 1:
        ys = ys[:, 0]
    if preds.ndim > 1:
        preds = preds[:, 0]
    if vars_.ndim > 1:
        vars_ = vars_[:, 0]

    # Sort for a smooth curve
    idx = torch.argsort(xs)

    xs = xs[idx]
    ys = ys[idx]
    preds = preds[idx]
    vars_ = vars_[idx]

    std = torch.sqrt(torch.clamp(vars_, min=0))

    plt.figure(figsize=(8, 5))

    # Keep only the first dimension of xs if it has more than one dimension
    if xs.ndim > 1:
        xs = xs[:, 0, 0].squeeze(0)
        print(xs.shape)
    if ys.ndim > 1:
        ys = ys[:, 0].squeeze(0)
        print(ys.shape)

    # Predictive mean
    plt.plot(xs, preds, color="#8bc34a", linewidth=2, label="Predictive mean")

    # ±1 standard deviation
    plt.fill_between(
        xs.numpy(),
        (preds - std).numpy(),
        (preds + std).numpy(),
        color="#c5e1a5",
        alpha=0.5,
        label=r"$\pm1\sigma$",
    )

    # Ground truth
    plt.scatter(xs, ys, s=15, color="#2e7d4f", label="Ground truth")

    # Uncomment for a 95% interval (approximately)
    # plt.fill_between(
    #     xs.numpy(),
    #     (preds - 2 * std).numpy(),
    #     (preds + 2 * std).numpy(),
    #     color="red",
    #     alpha=0.15,
    #     label=r"$\pm2\sigma$",
    # )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model prediction with predictive uncertainty")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(path, dpi=600)
    plt.show()

def plot_gumbel_softmax_approximation(N : int = 10, p : float = 0.2, n_samples : int = 1000, taus : list[float] = [0.01, 0.1, 1, 10] ):
    """
    Plot the Binomial distribution and above the plot the results of the gumbel softmax reparameterization trick above it.
    """

    binom = Binomial(total_count=N, probs=p)

    # Sample from the binomial distribution
    x = torch.arange(0, N + 1)
    logits = binom.log_prob(x)
    y = logits.exp()

    plt.figure(figsize=(8, 5), dpi=600)
    plt.bar(x.numpy(), y.numpy(), width=0.5, alpha=0.5, label="Binomial Distribution", color="#8bc34a")

    # Gumbel noise
    u = torch.rand(n_samples, N + 1)
    gumbel_samples = -torch.log(-torch.log(u + 1e-10))

    # alphas just for plotting
    alphas = [1, 0.75, 0.5, 0.35]
    for tau, alpha in zip(taus, alphas):
        gumbel_softmax_samples = torch.softmax((logits + gumbel_samples) / tau, dim=-1)
        approx_distribution = gumbel_softmax_samples.mean(dim=0)
        plt.plot(x.numpy(), approx_distribution.numpy(), label=f"Gumbel-Softmax (τ={tau})", linewidth=2, color="#2e7d4f", alpha = alpha)

    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Gumbel-Softmax Approximation of Binomial Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig("img/gumbel_softmax_approximation.png", dpi=600)

    plt.show()



def plot_gaussian_approximation(N : int = 100, p : float = 0.2,  n_samples : int = 1000, bins : int = 100):
    """
    Plot the Gaussian reparameterization trick
    """
    binom = Binomial(total_count=N, probs=p)
    mu = N * p
    sigma = math.sqrt(N * p * (1 - p))
    normal = Normal(loc=mu, scale=sigma)

    x = torch.arange(0, N + 1)
    logits = binom.log_prob(x)
    y = logits.exp()

    plt.figure(figsize=(8, 5), dpi=600)
    plt.bar(x.numpy(), y.numpy(), width=0.5, alpha=0.85, label="Binomial Distribution", color="#8bc34a")


    # Sample from the normal distribution
    norm_x = torch.linspace(mu - 4 * sigma, mu + 4 * sigma, n_samples)
    norm_y = normal.log_prob(norm_x).exp()

    plt.plot(norm_x.numpy(), norm_y.numpy(), label="Normal Distribution", linewidth=2, color="#2e7d4f")

    z = torch.randn(n_samples)
    rep_samples = mu + sigma * z
    plt.hist(rep_samples.numpy(), bins=bins, density=True, alpha=0.5,
             color="#c5e1a5", label="Reparameterized Samples")

    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Gaussian Approximation of Binomial Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig("img/gaussian_approximation.png", dpi=600)

    plt.show()




if __name__ == "__main__":
    # Example usage of plot_gumbel_softmax_approximation
    plot_gumbel_softmax_approximation(N = 15, p = 0.2, n_samples = 1000, taus = [0.1, 1, 10, 1000])
    plot_gaussian_approximation(N = 15, p = 0.2, n_samples = 1000, bins = 100)
