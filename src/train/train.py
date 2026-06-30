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
from src.vis.vis import *

def train(model : BaseMLP, loader : DataLoader, optimizer : Optimizer, criterion : F, device : str = "cuda" if torch.cuda.is_available() else "cpu"):
    model.train()

    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        reg_loss = 0.0
        if hasattr(model, "regularization_loss"):
            reg_loss = model.regularization_loss() # If needed this can be used

        loss = loss + reg_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model : BaseMLP, loader : DataLoader, criterion : F, device : str = "cuda" if torch.cuda.is_available() else "cpu"):

    model.eval()

    total_loss = 0.0

    for x, y in loader:

        x, y = x.to(device), y.to(device)

        out = model(x)

        loss = criterion(out, y) + model.regularization_loss()

        total_loss += loss.item()

    return total_loss / len(loader)

def fit(model : BaseMLP, train_loader : DataLoader, val_loader : DataLoader, optimizer : Optimizer, criterion : F, epochs : int = 1000, device : str = "cuda" if torch.cuda.is_available() else "cpu"):

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

    n_samples = 1000

    input_dim = 1
    output_dim = 1

    x = torch.randn(n_samples, input_dim)
    y = torch.sin(x) + 0.5 * torch.sin(2 * x) + 0.25 * x ** 2 # sum of sins toy function
    
    dataset = TensorDataset(x, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    batch_size = 32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"input_dim" : input_dim, "output_dim" : output_dim, "device" : device}

    model = BGA_MLP(cfg)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = 1e-4)

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=100,
        device=device,
    )

    plot_history(history)
    plot_results(model, val_loader, device)
