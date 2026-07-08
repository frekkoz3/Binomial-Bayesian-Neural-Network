"""
██████  ██████  ██████  ███    ██
     ██ ██   ██      ██ ████   ██
 █████  ██████   █████  ██ ██  ██
██      ██   ██ ██      ██  ██ ██
███████ ██████  ███████ ██   ████
"""

from torch.optim import Optimizer, Adam, AdamW
import argparse
import yaml
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.net import *
from src.net.mlp import *
from src.net.g_mlp import *
from src.net.bga_mlp import *
from src.net.bgs_mlp import *
from src.data.data import *
from src.vis.vis import *
from src.train.train import *



def plot_histories(histories, names, save, path):
    """
    Plot several training histories in the same figure.
    One figure for the training losses, one for the validation losses.

    Path should be without .PNG extension
    """
    alphas = [0.25, 0.5, 0.65, 0.85]

    # Training Losses
    plt.figure(figsize=(8, 5))

    for i, history in enumerate(histories):
        plt.plot(history["train_loss"], label=f"Train - {names[i]}", color="#2e7d4f", alpha=alphas[i])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training History")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    if save:
        img_path = path + ".png"
        plt.savefig(img_path, dpi=600)

    # Validation Losses
    plt.figure(figsize=(8, 5))

    for i, history in enumerate(histories):
        plt.plot(history["val_loss"], label=f"Val - {names[i]}", color="#2e7d4f", alpha=alphas[i])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Validation History")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    if save:
        img_path = path + "_val.png"
        plt.savefig(img_path, dpi=600)



def plot_predictions(models, val_loader, device, names, save, path):
    """Plot predictions for models, split into two figures (1st & 3rd, and 2nd & 4th)."""

    # Ensure we have exactly 4 models for this specific grouping logic
    if len(models) != 4:
        raise ValueError(f"Expected 4 models, but got {len(models)}.")

    # Groupings: (Figure Number, [Model Indices])
    # Indices are 0-based: [0, 2] = 1st and 3rd; [1, 3] = 2nd and 4th
    figure_groups = [
        (1, [0, 2]),
        (2, [1, 3])
    ]

    # Color Palette: (Mean color, Std fill color)
    # 1st model in plot: Dark Green. 2nd model in plot: Light Green.
    palette = [
        ("#2e7d32", "#66bb6a"),  # Dark green line, lighter green fill
        ("#7cb342", "#c5e1a5")   # Light green line, lighter green fill
    ]
    gt_color = "#1b5e20"         # Very dark green for ground truth

    for fig_num, model_indices in figure_groups:
        plt.figure(figsize=(16, 9))

        # Variables to store ground truth so we only plot it once per figure
        xs_gt, ys_gt = None, None

        for color_idx, m_idx in enumerate(model_indices):
            model = models[m_idx]
            model.eval()

            xs, ys, preds, vars_ = [], [], [], []

            # Use no_grad for evaluation to save memory and compute
            with torch.no_grad():
                for x, y in val_loader:
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

            # Concatenate and squeeze
            xs = torch.cat(xs).squeeze()
            ys = torch.cat(ys).squeeze()
            preds = torch.cat(preds).squeeze()
            vars_ = torch.cat(vars_).squeeze()

            # Ensure tensors are 1D
            if xs.ndim > 1: xs = xs[:, 0]
            if ys.ndim > 1: ys = ys[:, 0]
            if preds.ndim > 1: preds = preds[:, 0]
            if vars_.ndim > 1: vars_ = vars_[:, 0]

            # Sort for a smooth curve
            idx = torch.argsort(xs)
            xs = xs[idx]
            ys = ys[idx]
            preds = preds[idx]
            vars_ = vars_[idx]

            std = torch.sqrt(torch.clamp(vars_, min=0))

            # Capture ground truth from the first model iteration in this figure
            if xs_gt is None:
                xs_gt, ys_gt = xs, ys

            mean_color, std_color = palette[color_idx]

            # Predictive mean
            plt.plot(
                xs, preds, color=mean_color, linewidth=2,
                label=f"Pred mean - {names[m_idx]} init"
            )

            # ±1 standard deviation (using lighter color and slow alpha)
            plt.fill_between(
                xs.numpy(),
                (preds - std).numpy(),
                (preds + std).numpy(),
                color=std_color,
                alpha=0.3,
                label=rf"$\pm1\sigma$ - {names[m_idx]}"
            )

        # Ground truth (plotted last with high zorder so it sits on top)
        plt.scatter(
            xs_gt, ys_gt, s=15, color=gt_color,
            label="Ground truth", zorder=5
        )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Models prediction - Group {fig_num}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Handle saving securely without overwriting
        if save:
            p = Path(path)
            # Example: "results.png" becomes "results_fig1.png"
            save_path = p.with_name(f"{p.stem}_fig{fig_num}{p.suffix}")
            plt.savefig(save_path, dpi=600)

        plt.show()



if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save = False
    # Setup some parameters
    n_samples = 1000
    batch_size = 32
    input_dim = 1
    output_dim = 1

    train_prop = 0.8
    val_prop = 0.2
    shuffle = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "xpu" if torch.xpu.is_available() else device

    n_epochs = 10

    name_models = ["BGA_MLP", "BGS_MLP"]

    histories = []
    models = []

    for mode in ["with", "without"]:
        for model_name in name_models:
            # Move everything to config
            cfg = {"model": model_name,
                   "dataset": "SinusoidData",
                   "n_samples" : n_samples,
                   "input_dim" : input_dim,
                   "output_dim" : output_dim,
                   "train_prop": train_prop,
                   "val_prop": val_prop,
                   "batch_size": batch_size,
                   "shuffle": shuffle,
                   "device" : device,
                   "weight_init" : True if mode == "with" else False,
                   "tau_scheduler" : "ConstantTauScheduler",
                   "bias" : True,

                   "lower_bound" : -1.5,
                   "upper_bound" : 1.5,
                   "minimum": -2,
                   "maximum": 2
                   }

            # Get data
            dataset_loader = eval(cfg["dataset"])(**cfg)
            train_loader, val_loader, test_loader, _, _, _ = dataset_loader.generate_data(**cfg)

            # Load model
            model = eval(cfg["model"])(cfg)
            model.to(device)

            criterion = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr = 1e-2)

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

            run_folder = f"models/impact_init_{timestamp}/{mode}_{model_name.lower()}"
            if not os.path.exists(run_folder):
                os.makedirs(run_folder, exist_ok=True)

            histories.append(history)
            models.append(model)

            torch.save(
                model.state_dict(),
                f"{run_folder}/model.pt"
            )

            with open(f"{run_folder}/config.yaml", "w") as f:
                yaml.dump(cfg, f)

    names = ["BGA with", "BGS with", "BGA without", "BGS without"]
    run_folder = f"models/impact_init_{timestamp}"
    plot_histories(histories=histories, names=names, save=True, path=f"{run_folder}/histories")
    plot_predictions(models=models, val_loader=val_loader, device=device, names=names, save=True, path=f"{run_folder}/predictions.png")