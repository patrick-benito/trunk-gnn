import os
from datetime import datetime
from argparse import Namespace
import wandb
import yaml
import platform

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
        args.model = config.model
        args.notes = ""

        train.main(args)


def main():
    sweep_config = load_sweep_config("config/sweep_config.yaml")
    is_iris = platform.node() == "iris"
    local = "" if is_iris else "local-"
    project_name = f"{local}trunk-{sweep_config['parameters']['model']['values'][0]}-0.1.0-sweep"
    
    if is_iris:
        commit_message = os.popen('git log -1 --pretty=%B').read().strip()
        sweep_config['name'] = commit_message.replace('[TRAIN]', '').strip()
        print(f"Commit message: {sweep_config['name']}")

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    main()
