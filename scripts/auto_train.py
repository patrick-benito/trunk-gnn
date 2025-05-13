import os
from datetime import datetime
from argparse import Namespace
import wandb
import yaml
import platform

import train as train
from trunk_gnn.train_utils import setup_config, load_config

def train_sweep(config=None):
    # override args with config for all keys in config

    with wandb.init(config=config):
        full_config = setup_config(train.get_args())

        for key, value in wandb.config.items():
            full_config[key] = value

        full_config['save_model'] = True
        wandb.config.update(full_config)

        train.main()

def main():
    sweep_config = load_config("config/sweep_config.yaml")
    is_iris = platform.node() == "iris"
    local = "" if is_iris else "local-"
    project_name = f"{local}trunk-{sweep_config['parameters']['model']['values'][0]}-0.2.1-sweep"
    
    if is_iris:
        commit_message = os.popen('git log -1 --pretty=%B').read().strip()
        sweep_config['name'] = commit_message.replace('[TRAIN]', '').strip()

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    main()
