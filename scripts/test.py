import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import argparse
import os

from trunk_gnn.data import TrunkGraphDataset
from trunk_gnn.model import TrunkGNN, TrunkMLP

from trunk_gnn.train_utils import init_wandb, set_seed, epoch, save_model
from trunk_gnn.dataset_utils import dataset_split
from trunk_gnn.test_utils import open_loop_test_all, load_data_sets_from_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_artifacts(artifact_name):
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    artifact.download(root="artifacts")

def main(args):
    if args.artifact_name:
        download_artifacts(args.artifact_name)

    model_data = torch.load(os.path.join(args.artifacts_folder, "model_data.pth"))
    wandb.init(mode="disabled", config=model_data['config'])
    print(f"Initialized wandb with config: {wandb.config}")

    model = TrunkGNN()
    model.load_state_dict(torch.load(os.path.join(args.artifacts_folder, "model_state_dict.pth")))
    model.to(device)
    
    with torch.no_grad():
        open_loop_test_all(model, load_data_sets_from_folder(args.test_dataset_folder), save_figures=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_folder", type=str, default="./artifacts/", help="Path to the artifacts folder.")
    parser.add_argument(
        "--test_dataset_folder",
        type=str,
        default="./data/mass_100g_harmonic/test/",
        help="Path of the testing dataset folders"
    )
    parser.add_argument("--artifact_name", type=str, default=None, help="Artifact name to download from wandb.")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
