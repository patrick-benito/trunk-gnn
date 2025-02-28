import os
from datetime import datetime
from argparse import Namespace
import wandb
import yaml

import train


def load_sweep_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train.main(
            Namespace(
                data_set_folder="data/sim_data/n_body_test_1",
                infer_data_set_folder="data/sim_data/n_body_test_1",
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                gradient_clipping_max_norm=config.gradient_clipping_max_norm,
                normalization_strategy="none",
                seed=0,
                wandb=True,
                notes="",
                save_model=True,
                infer=False,
                scheduler_patience=config.scheduler_patience,
            )
        )


def main():
    sweep_config = load_sweep_config("config/sweep_config.yaml")
    sweep_id = wandb.sweep(sweep_config, project="GNN-Sweep-Test")
    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    main()
