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
    parser = argparse.ArgumentParser(description="Train a model on the UCI regression dataset.")
    parser.add_argument("--dataset-id", type=int, default=165, help="UCI Dataset to use")
    args = parser.parse_args()

    dataset_id = args.dataset_id

    # Setup some parameters
    n_samples = 1000
    batch_size = 16
    input_dim = 1
    output_dim = 1

    train_prop = 0.8
    val_prop = 0.2
    shuffle = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "xpu" if torch.xpu.is_available() else device

    n_epochs = 1

    models = ["BGS_MLP", "BGA_MLP", "G_MLP", "MLP"]

    for model_name in models:
        # Move everything to config
        cfg = {"model": model_name,
               "dataset": "UCIRegressionDataset",
               "train_prop": train_prop,
               "val_prop": val_prop,
               "batch_size": batch_size,
               "shuffle": shuffle,
               "device" : device,
               "tau_scheduler" : "ConstantTauScheduler",
               "bias" : True,

               "lower_bound" : -1.5,
               "upper_bound" : 1.5,

               "uci_id" : 165
               }

        # Get data
        dataset_loader = eval(cfg["dataset"])(**cfg)
        train_loader, val_loader, test_loader = dataset_loader.generate_data()

        input_dim, output_dim = dataset_loader._get_dims()
        cfg["input_dim"] = input_dim
        cfg["output_dim"] = output_dim

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

        # Plot results
        plot_history(history, model_name, save=True)

        # get timestamp for saving
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

        torch.save(model.state_dict(), f"models/{model_name.lower()}/uci_{dataset_id}_{timestamp}.pt")
        # save config in the same folder:
        with open(f"models/{model_name.lower()}/uci_{dataset_id}_{timestamp}.yaml", "w") as f:
            yaml.dump(cfg, f)

