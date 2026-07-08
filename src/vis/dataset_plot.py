r"""
██████  ██████  ██████  ███    ██
     ██ ██   ██      ██ ████   ██
 █████  ██████   █████  ██ ██  ██
██      ██   ██ ██      ██  ██ ██
███████ ██████  ███████ ██   ████
"""

import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.optim import Adam
import numpy as np

from src.data.data import UCI_DATASETS, UCIRegressionDataset
from src.net.bga_mlp import BGA_MLP
from src.net.bgs_mlp import BGS_MLP
from src.net.g_mlp import G_MLP
from src.net.mlp import MLP
from src.train.train import fit
from src.val.metrics import gaussian_nll, rmse

def plot_dataset_vs_metric(results: dict[str, list[dict]], metric: str, save_path: str):
    # Collect all unique datasets, preserving insertion order
    dataset_names = list(dict.fromkeys(row["dataset"] for rows in results.values() for row in rows))
    models = list(results.keys())

    x = np.arange(len(dataset_names))
    width = 0.8 / len(models)  # total group width = 0.8

    plt.figure(figsize=(8, 5))

    colors = [ "#2e7d4f", "#c5e1a5", "#8bc34a", "#5e910d"]

    for i, model in enumerate(models):
        # Map dataset -> metric
        metric_map = {row["dataset"]: row[metric] for row in results[model]}
        values = [metric_map.get(d, np.nan) for d in dataset_names]

        plt.bar(
            x + (i - (len(models) - 1) / 2) * width,
            values,
            width=width,
            label=model,
            color=colors[i],
        )

    plt.xticks(x, dataset_names, rotation=20, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel(metric.upper())
    plt.yscale("log")
    plt.title(f"Dataset vs {metric.upper()}, by model")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {"rmse": rmse, "nll": gaussian_nll}
    datasets = ("concrete", "energy", "airfoil")
    epochs = 500

    results = {}

    # Dataset loader loop
    for ds_name in datasets:
        dataset = UCIRegressionDataset(UCI_DATASETS[ds_name], batch_size=64)
        train_loader, val_loader, test_loader = dataset.generate_data()

        shared_cfg = {"input_dim": dataset.input_dim, "output_dim": dataset.output_dim*2, "hidden_dims": [20, 20], "n_hidden_layer": 2, "heteroscedastic": True}

        # Model loop
        for name, cls, extra_cfg in [
            ("MLP", MLP, {}),
            ("G_MLP", G_MLP, {}),
            ("BGA_MLP", BGA_MLP, {"N": 50}),
            ("BGS_MLP", BGS_MLP, {"N": 50, "tau_scheduler": "ExponentialTauScheduler", "device": device}),
        ]:
            model = cls({**shared_cfg, **extra_cfg}).to(device)

            fit(model, train_loader, val_loader,
                optimizer=Adam(model.parameters(), lr=1e-3),
                criterion=nn.GaussianNLLLoss(full=True), epochs=epochs, device=device)

            results.setdefault(name, []).append({
                "dataset": ds_name,
                **{m_name: metric(model, test_loader, device) for m_name, metric in metrics.items()}
            })

    plot_dataset_vs_metric(results, metric="rmse", save_path="plots/dataset_vs_rmse.png")
    plot_dataset_vs_metric(results, metric="nll", save_path="plots/dataset_vs_nll.png")