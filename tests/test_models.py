import argparse
import torch
from torch_geometric.data import Data
import pytest
import wandb
import os
from torch_geometric.loader import DataLoader

from trunk_gnn.model import TrunkGNN, TrunkMLP
from trunk_gnn.data import TrunkGraphDataset
from trunk_gnn.train_utils import init_wandb, setup_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def test_gnn():
    init_wandb(setup_config(get_args()))
    model = TrunkGNN().to(device)
    small_set = os.path.join(wandb.config["dataset_folder"],"test", "1")
    dataset = TrunkGraphDataset(small_set, device=device)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    model.eval()

    for data in dataloader:
        assert model(data) is not None

if __name__ == "__main__":
    test_gnn()