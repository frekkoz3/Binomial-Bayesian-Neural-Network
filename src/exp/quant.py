r"""
██████  ██████  ██████  ███    ██
     ██ ██   ██      ██ ████   ██
 █████  ██████   █████  ██ ██  ██
██      ██   ██ ██      ██  ██ ██
███████ ██████  ███████ ██   ████
"""

import copy

import torch
from torch import nn
from torch.optim import Adam

from src.data.data import UCI_DATASETS, UCIRegressionDataset
from src.net.bga_mlp import BGA_MLP
from src.train.train import fit
from src.val.metrics import rmse


def resolution_foreach(model, test_loader, device, bit_widths=(32, 16, 8, 6, 4, 2), metrics={"rmse": rmse}):
    """
    Evaluate a quantized copy of `model` at each bit-width in `bit_widths`.
    `model` itself (and its full-precision `rho`) is left untouched.
    """
    results = []

    for bits in bit_widths:

        # Re-quantization
        m = copy.deepcopy(model)
        m.set_resolution(bits)
        state = m.export_inference_state(quantize=True)
        m.load_inference_state(state)

        results.append({
            "bits": bits,
            **{name: metric(m, test_loader, device) for name, metric in metrics.items()}
        })

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = UCIRegressionDataset(UCI_DATASETS["concrete"], batch_size=64)
    train_loader, val_loader, test_loader = dataset.generate_data()

    cfg = {"input_dim": dataset.input_dim, "output_dim": dataset.output_dim,
           "hidden_dims": 50, "N": 50}
    model = BGA_MLP(cfg).to(device)

    fit(model, train_loader, val_loader,
        optimizer=Adam(model.parameters(), lr=1e-3),
        criterion=nn.MSELoss(),
        epochs=4000,
        device=device)

    print(f"{'bits':>4}  {'rmse':>8}")
    for row in resolution_foreach(model, test_loader, device):
        print(f"{row['bits']:>4}  {row['rmse']:>8.4f}")
