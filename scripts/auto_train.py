import os
from datetime import datetime
from argparse import Namespace
import wandb
import yaml

import train as train


def load_sweep_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        args = train.get_args()

        args.num_epochs = config.num_epochs
        args.batch_size = config.batch_size
        args.learning_rate = config.learning_rate
        args.weight_decay = config.weight_decay
        args.gradient_clipping_max_norm = config.gradient_clipping_max_norm
        args.scheduler_patience = config.scheduler_patience
        args.save_model = True
        args.wandb = True
        args.seed = 0
        args.notes = ""


        train.main(args)


def main():
    sweep_config = load_sweep_config("config/sweep_config.yaml")
    sweep_id = wandb.sweep(sweep_config, project="trunk-gnn-0.1.0-sweep")
    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    main()
