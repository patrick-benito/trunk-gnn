import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import argparse
import os

from trunk_gnn.data import TrunkGraphDataset
from trunk_gnn.model import TrunkGNN, TrunkMLP

from trunk_gnn.train_utils import init_wandb, set_seed, epoch, save_model, setup_config
from trunk_gnn.dataset_utils import dataset_split
from trunk_gnn.test_utils import open_loop_test_all, load_data_sets_from_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def train(
    model,
    criterion,
    optimizer,
    train_data_loader,
    validation_data_loader,
    test_data_loader,
    open_loop_test_data_sets,
):
    set_seed(wandb.config["seed"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=wandb.config["scheduler_patience"]
    )

    model.to(device)

    for i in tqdm(range(wandb.config["num_epochs"])):
        epoch(
            model,
            optimizer,
            criterion,
            scheduler,
            train_data_loader,
            validation_data_loader,
            test_data_loader,
            device,
            gradient_clipping_max_norm=wandb.config["gradient_clipping_max_norm"],
        )

        if i % 10 == 0:
            open_loop_test_all(model, open_loop_test_data_sets)
        
    open_loop_test_all(model, open_loop_test_data_sets)

def main():
    print(f"Initialized wandb with config: {wandb.config}")

    dataset = TrunkGraphDataset(os.path.join(wandb.config["dataset_folder"],"train"), device=device, link_step=wandb.config["link_step"])
    train_data_set, validation_data_set, test_data_set = dataset_split(
        dataset, [0.72, 0.14, 0.14]
    )
    
    if wandb.config["model"] == "gnn":
        print("Using GNN model")
        model = TrunkGNN()
    elif wandb.config["model"] == "mlp":
        print("Using MLP model")
        model = TrunkMLP()
        assert wandb.config["link_step"] == 1, "MLP model only supports link_step=1"
        assert wandb.config["shuffle"] == False, "MLP model only supports shuffle=False"
    else:
        raise ValueError(f"Model {wandb.config['model']} not supported.")
    
    train(
        model,
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(
            model.parameters(), lr=wandb.config["learning_rate"], weight_decay=wandb.config["weight_decay"]
        ),
        train_data_loader=DataLoader(
            train_data_set, batch_size=wandb.config["batch_size"], shuffle=wandb.config["shuffle"]
        ),
        validation_data_loader=DataLoader(
            validation_data_set, batch_size=wandb.config["batch_size"], shuffle=wandb.config["shuffle"]
        ),
        test_data_loader=DataLoader(
            test_data_set, batch_size=wandb.config["batch_size"], shuffle=wandb.config["shuffle"]
        ),
        open_loop_test_data_sets=load_data_sets_from_folder(os.path.join(wandb.config["dataset_folder"],"test"), link_step=wandb.config["link_step"]),
    )

    if wandb.config["save_model"]:
        save_model(model, dataset)

    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model to train.")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="data/mass_100g_harmonic/",
        help="Path of the training dataset file.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for optimization.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--notes", type=str, default="", help="Save notes")
    parser.add_argument("--save_model", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    init_wandb(setup_config(get_args()))
    main()
