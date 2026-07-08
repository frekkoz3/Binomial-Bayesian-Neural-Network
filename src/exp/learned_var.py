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

from src.net import *
from src.net.mlp import *
from src.net.g_mlp import *
from src.net.bga_mlp import *
from src.net.bgs_mlp import *
from src.data.data import *
from src.vis.vis import *
from src.train.train import *


@torch.no_grad()
def plot_variances(models: list, dataloader, names: list, device="cpu", save=False, path=None, n_samples=100):
    """
    Plots predictive uncertainty for multiple models on side-by-side subplots with strict font scaling.
    """
    # 1. Collect and sort dataset (shared across all models)
    xs, ys = [], []
    for x, y in dataloader:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs).squeeze().cpu()
    ys = torch.cat(ys).squeeze().cpu()

    idx = torch.argsort(xs)
    xs, ys = xs[idx], ys[idx]

    n_models = len(models)
    # Adjusted figsize to be slightly wider per subplot for better text fitting
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True, dpi=300)

    if n_models == 1:
        axes = [axes]

    for i, (model, ax, name) in enumerate(zip(models, axes, names)):
        # 2. Compute mean theoretical variance E[V[w]]
        vars_list = []
        for module in model.modules():
            rho = getattr(module, 'rho', getattr(module, 'weight_rho', None))
            if isinstance(rho, torch.nn.Parameter):
                p = torch.sigmoid(rho)
                vars_list.append((module.N * p * (1 - p)).mean().item())

        mean_var = sum(vars_list) / len(vars_list) if vars_list else 0.0

        # 3. Monte Carlo predictions
        model.train() # Force stochasticity on
        preds = torch.stack([model(xs.unsqueeze(1).to(device)).squeeze().cpu() for _ in range(n_samples)])
        model.eval()  # Restore state

        y_mean = preds.mean(dim=0)
        y_std = preds.std(dim=0)

        # 4. Plotting
        ax.fill_between(xs.numpy(), (y_mean - y_std).numpy(), (y_mean + y_std).numpy(),
                        color="#c5e1a5", alpha=0.5, label=r"$\pm1\sigma$")

        ax.plot(xs.numpy(), y_mean.numpy(), color="#8bc34a", linewidth=2, label="Predictive mean")

        # Reduced scatter size (s=8) to prevent visual clutter
        ax.scatter(xs.numpy(), ys.numpy(), s=8, color="#2e7d4f", label="Ground truth", zorder=3)

        # # 5. Strictly controlled Text and Label sizes
        # text_str = r"$\mathbb{E}[\mathbb{V}\text{ }[w^{(i)}]] = " + f"{mean_var:.2f}$"
        # ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#2e7d4f'))

        ax.set_xlabel("x", fontsize=11)
        if i == 0:
            ax.set_ylabel("y", fontsize=11)

        ax.set_title(name, fontsize=13, pad=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)

    # Use bbox_inches='tight' to ensure nothing is cropped out during save
    plt.tight_layout()
    if save and path:
        plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save = False
    # Setup some parameters
    n_samples = 10000
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

    choices = torch.tensor(np.arange(-2, 2.1, 0.1), dtype=torch.float32)

    histories = []
    models = []

    for model_name in name_models:
        # Move everything to config
        cfg = {"model": model_name,
               "dataset": "DiscreteNoisyDataset",
               "n_samples" : n_samples,
               "input_dim" : input_dim,
               "output_dim" : output_dim,
               "train_prop": train_prop,
               "val_prop": val_prop,
               "batch_size": batch_size,
               "shuffle": shuffle,
               "device" : device,
               "weight_init" : True,
               "tau_scheduler" : "ConstantTauScheduler",
               "bias" : True,

               "lower_bound" : -1.5,
               "upper_bound" : 1.5,
               "minimum": -2,
               "maximum": 2,
               "sigma_squared":0.5,
               "choices": choices
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

        run_folder = f"models/learned_var_{timestamp}/{model_name.lower()}"
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
    run_folder = f"models/learned_var_{timestamp}"
    plot_variances(
        models=models,
        dataloader=val_loader,
        names=names,
        device=device,
        save=True,
        path=f"{run_folder}/predictions.png"
    )