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
# these functions are vibe coded

def plot_history(history: dict):
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

    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_results(model, dataloader, device="cpu"):
    """
    Plot predictions for a 1D regression problem.

    Assumes:
        input_dim = 1
        output_dim = 1
    """

    model.eval()

    xs = []
    ys = []
    preds = []

    for x, y in dataloader:

        x = x.to(device)

        pred = model(x)

        xs.append(x.cpu())
        ys.append(y.cpu())
        preds.append(pred.cpu())

    xs = torch.cat(xs).squeeze()
    ys = torch.cat(ys).squeeze()
    preds = torch.cat(preds).squeeze()

    # sort so the prediction is a proper curve
    idx = torch.argsort(xs)

    xs = xs[idx]
    ys = ys[idx]
    preds = preds[idx]

    plt.figure(figsize=(8, 5))

    plt.scatter(xs, ys, s=15, label="Ground truth")
    plt.plot(xs, preds, linewidth=2, label="Prediction")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model prediction")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()



def plot_gumbel_softmax_approximation(N : int = 10, p : float = 0.2, n_samples : int = 1000, taus : list[float] = [0.01, 0.1, 1, 10] ):
    """
    Plot the Binomial distribution and above the plot the results of the gumbel softmax reparameterization trick above it.
    """

    from torch.distributions import Binomial

    binom = Binomial(total_count=N, probs=p)

    # Sample from the binomial distribution
    x = torch.arange(0, N + 1)
    logits = binom.log_prob(x)
    y = logits.exp()

    plt.figure(figsize=(8, 5))
    plt.bar(x.numpy(), y.numpy(), width=0.5, alpha=0.5, label="Binomial Distribution")

    # Gumbel noise
    u = torch.rand(n_samples, N + 1)
    gumbel_samples = -torch.log(-torch.log(u + 1e-10))

    for tau in taus:
        gumbel_softmax_samples = torch.softmax((logits + gumbel_samples) / tau, dim=-1)
        approx_distribution = gumbel_softmax_samples.mean(dim=0)
        plt.plot(x.numpy(), approx_distribution.numpy(), label=f"Gumbel-Softmax (τ={tau})", linewidth=2)

    plt.xlabel("x")
    plt.ylabel("Probability")
    plt.title("Gumbel-Softmax Approximation of Binomial Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # Example usage of plot_gumbel_softmax_approximation
    plot_gumbel_softmax_approximation(N=10, p=0.2, n_samples=1000, taus=[0.01, 0.1, 1, 10, 1000])
