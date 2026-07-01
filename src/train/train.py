r"""
██████  ██████  ██████  ███    ██ 
     ██ ██   ██      ██ ████   ██ 
 █████  ██████   █████  ██ ██  ██ 
██      ██   ██ ██      ██  ██ ██ 
███████ ██████  ███████ ██   ████                                 
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Optimizer, Adam, AdamW

from tqdm import tqdm

from src.net import *
from src.net.mlp import *
from src.net.g_mlp import *
from src.net.bga_mlp import *
from src.net.bgs_mlp import *
from src.data.data import *
from src.vis.vis import *


def train(model : BaseMLP,
          loader : DataLoader,
          optimizer : Optimizer,
          criterion : F,
          device : str = "cuda" if torch.cuda.is_available() else "cpu"):
    model.train()

    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        reg_loss = model.regularization_loss() if hasattr(model, "regularization_loss") else 0.0
        loss = loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model : BaseMLP,
             loader : DataLoader,
             criterion : F,
             device : str = "cuda" if torch.cuda.is_available() else "cpu"):

    model.eval()

    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)

        reg_loss = model.regularization_loss() if hasattr(model, "regularization_loss") else 0.0
        loss = criterion(out, y) + reg_loss
        total_loss += loss.item()

    return total_loss / len(loader)


def fit(model : BaseMLP,
        train_loader : DataLoader,
        val_loader : DataLoader,
        optimizer : Optimizer,
        criterion : F,
        epochs : int = 1000,
        device : str = "cuda" if torch.cuda.is_available() else "cpu"):

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return history



if __name__ == '__main__':

    # Setup some parameters
    n_samples = 1000
    batch_size = 32
    input_dim = 1
    output_dim = 1

    train_prop = 0.8
    val_prop = 0.2
    shuffle_data = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "xpu" if torch.xpu.is_available() else device

    n_epochs = 100

    # Move everything to config
    cfg = {"model": "BGS_MLP",
           "dataset": "SinusoidData",
           "n_samples" : n_samples,
           "input_dim" : input_dim,
           "output_dim" : output_dim,
           "train_prop": train_prop,
           "val_prop": val_prop,
           "batch_size": batch_size,
           "shuffle_data": shuffle_data,
           "device" : device,
           "tau_scheduler" : "ExponentialTauScheduler"
           }

    # Get data
    dataset_loader = eval(cfg["dataset"])(cfg["n_samples"],
                                          cfg["train_prop"],
                                          cfg["val_prop"],
                                          cfg["batch_size"],
                                          cfg["input_dim"],
                                          cfg["shuffle_data"])
    train_loader, val_loader, test_loader, _ , _, _ = dataset_loader.generate_data()

    # Load model
    model = eval(cfg["model"])(cfg)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = 1e-4)

    # Fit model
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=n_epochs,
        device=device,
    )

    # Plot results
    plot_history(history)
    plot_results(model, val_loader, device)
