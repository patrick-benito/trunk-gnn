import argparse
import torch

from trunk_gnn.model import TrunkGNN, TrunkMLP
from trunk_gnn.test_utils import open_loop_test_all, load_data_sets_from_folder, load_model

import sys
sys.path.append('./opt_ssm_model')

from ssm_torch_model import SSMModel
model = SSMModel(model_type="ssmr_orth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def main(args):
    #model = load_model(TrunkMLP, args.artifacts_folder, args.artifact_name)
    
    with torch.no_grad():
        open_loop_test_all(model, load_data_sets_from_folder(args.test_dataset_folder), save_figures=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_folder", type=str, default="./artifacts/", help="Path to the artifacts folder.")
    parser.add_argument(
        "--test_dataset_folder",
        type=str,
        default="./data/mass_500g_harmonic_ssm/test/",
        help="Path of the testing dataset folders"
    )
    parser.add_argument("--artifact_name", type=str, default=None, help="Artifact name to download from wandb.")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
