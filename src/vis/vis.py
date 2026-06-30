r"""
██████  ██████  ██████  ███    ██ 
     ██ ██   ██      ██ ████   ██ 
 █████  ██████   █████  ██ ██  ██ 
██      ██   ██ ██      ██  ██ ██ 
███████ ██████  ███████ ██   ████                                 
"""
import torch
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