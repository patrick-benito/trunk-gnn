import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import argparse

from trunk_gnn.data import TrunkGraphDataset

from trunk_gnn.model import TrunkGNN

from trunk_gnn.train_utils import init_wandb, set_seed, epoch, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
def train(
    args,
    model,
    criterion,
    optimizer,
    train_data_loader,
    validation_data_loader,
    test_data_loader,
):
    set_seed(args.seed)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=args.scheduler_patience
    )

    model.to(device)

    for _ in tqdm(range(args.num_epochs)):
        epoch(
            model,
            optimizer,
            criterion,
            scheduler,
            train_data_loader,
            validation_data_loader,
            test_data_loader,
            device,
            gradient_clipping_max_norm=args.gradient_clipping_max_norm,
        )


def main(args):
    init_wandb(args)

    dataset = TrunkGraphDataset(args.data_set_folder, device=device)
    train_data_set, validation_data_set, test_data_set = torch.utils.data.random_split(
        dataset, [0.72, 0.14, 0.14]
    )

    model = TrunkGNN(num_links=dataset.num_links)
    train(
        args,
        model,
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        ),
        train_data_loader=DataLoader(
            train_data_set, batch_size=args.batch_size, shuffle=True
        ),
        validation_data_loader=DataLoader(
            validation_data_set, batch_size=args.batch_size, shuffle=True
        ),
        test_data_loader=DataLoader(
            test_data_set, batch_size=len(test_data_set), shuffle=True
        ),
    )

    if args.save_model:
        save_model(model, dataset)

    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_set_folder",
        type=str,
        default="data/no_mass_100_train/",
        help="Path of the dataset file.",
    )

    parser.add_argument(
        "--num_epochs", type=int, default=2000, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimization.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_clipping_max_norm", type=float, default=1e6)
    parser.add_argument("--normalization_strategy", type=str, default="normalize")
    parser.add_argument("--scheduler_patience", type=str, default=40)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--notes", type=str, default="", help="Save notes")
    parser.add_argument("--save_model", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
