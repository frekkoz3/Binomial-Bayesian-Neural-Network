r"""
██████  ██████  ██████  ███    ██ 
     ██ ██   ██      ██ ████   ██ 
 █████  ██████   █████  ██ ██  ██ 
██      ██   ██ ██      ██  ██ ██ 
███████ ██████  ███████ ██   ████                                 
"""

"""
The goal of the experiment is to compare the behavior of B2N architectures on the extremes of the input space.
The dataset used here consists of samples from a sinusoidal function
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



if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save = True
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

    n_epochs = 100

    models = ["BGA_MLP", "BGS_MLP"]

    dt_type = "SinusoidDataset"

    for model_name in models:
        # Move everything to config
        cfg = {"model": model_name,
               "dataset": dt_type,
               "n_samples" : n_samples,
               "input_dim" : input_dim,
               "output_dim" : output_dim,
               "train_prop": train_prop,
               "val_prop": val_prop,
               "batch_size": batch_size,
               "shuffle": shuffle,
               "device" : device,
               "tau_scheduler" : "ConstantTauScheduler",
               "bias" : True,

               "lower_bound" : -1.5,
               "upper_bound" : 1.5,
               }

        # Get data
        dataset_loader = eval(cfg["dataset"])(**cfg)
        train_loader, val_loader, test_loader, _, _, _ = dataset_loader.generate_data()

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

        run_folder = f"models/tail_sinusoid_{timestamp}/{model_name.lower()}"
        if not os.path.exists(run_folder):
            os.makedirs(run_folder, exist_ok=True)

        plot_history(
            history,
            model_name,
            save=save,
            path=f"{run_folder}/history.png"
        )

        plot_results(
            model,
            val_loader,
            device=device,
            save=save,
            path=f"{run_folder}/results.png"
        )

        torch.save(
            model.state_dict(),
            f"{run_folder}/model.pt"
        )

        with open(f"{run_folder}/config.yaml", "w") as f:
            yaml.dump(cfg, f)

