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
from src.exp.quant import resolution_foreach
from src.net.bga_mlp import BGA_MLP
from src.net.bgs_mlp import BGS_MLP
from src.net.g_mlp import G_MLP
from src.net.mlp import MLP
from src.train.train import fit
from src.val.metrics import gaussian_nll, rmse

def plot_quantization_vs_metric(results: dict[str, list[dict]], metric: str, save_path: str):
    # Collect all unique bit-widths
    bit_widths = sorted({row["bits"] if row["bits"] else 32 for rows in results.values() for row in rows})
    models = list(results.keys())

    x = np.arange(len(bit_widths))
    width = 0.8 / len(models)  # total group width = 0.8

    plt.figure(figsize=(8, 5))

    colors = [ "#2e7d4f", "#c5e1a5", "#8bc34a"]

    for i, model in enumerate(models):
        if results[model][0]["bits"]:
            # Map bit-width -> metric
            metric_map = {row["bits"]: row[metric] for row in results[model]}
            values = [metric_map.get(b, np.nan) for b in bit_widths]

            plt.bar(
                x + (i - (len(models) - 1) / 2) * width,
                values,
                width=width,
                label=model,
                color=colors[i],
            )
        else:
            plt.axhline(results[model][0][metric], xmin=np.min(x), xmax=np.max(x), linestyle = "--", color=colors[i], label=f"Baseline - {model}")

    plt.xticks(x, bit_widths)
    plt.xlabel("Bits used to store p")
    plt.ylabel(metric.upper())
    plt.yscale("log")
    plt.title(f"Quantization bit-width vs {metric.upper()}, by model")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bit_widths = (32, 16, 8, 6, 4, 2)
    epochs = 500

    dataset = UCIRegressionDataset(UCI_DATASETS["energy"], batch_size=64)
    train_loader, val_loader, test_loader = dataset.generate_data()

    shared_cfg = {"input_dim": dataset.input_dim, "output_dim": dataset.output_dim*2, "hidden_dims": [20, 20], "n_hidden_layer": 2, "heteroscedastic": True}
    results = {}

    # Baseline: G_MLP
    for name, cls in [("G_MLP", G_MLP)]:
        model = cls(shared_cfg).to(device)
        fit(model, train_loader, val_loader,
            optimizer=Adam(model.parameters(), lr=1e-3),
            criterion=nn.GaussianNLLLoss(full=True), epochs=epochs, device=device)

        model_rmse = rmse(model, test_loader, device)
        model_nll = gaussian_nll(model, test_loader, device)
        results[name] = [{"bits": None, "rmse": model_rmse, "nll": model_nll}]

    # Quantizable models: BGA_MLP and BGS_MLP
    for name, cls, extra_cfg in [
        ("BGA_MLP", BGA_MLP, {"N": 50}),
        ("BGS_MLP", BGS_MLP, {"N": 50, "tau_scheduler": "ExponentialTauScheduler", "device": device}),
    ]:
        model = cls({**shared_cfg, **extra_cfg}).to(device)
        fit(model, train_loader, val_loader,
            optimizer=Adam(model.parameters(), lr=1e-3),
            criterion=nn.GaussianNLLLoss(full=True), epochs=epochs, device=device)

        results[name] = resolution_foreach(model, test_loader, device,
                                            bit_widths=bit_widths, metrics={"rmse": rmse, "nll": gaussian_nll})

    plot_quantization_vs_metric(results, metric="rmse", save_path="plots/quant_vs_rmse.png")
    plot_quantization_vs_metric(results, metric="nll", save_path="plots/quant_vs_nll.png")
