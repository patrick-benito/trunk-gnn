import wandb
import torch
import subprocess
import os
import shutil
from typing import Optional
import yaml
import json

def compute_l1_norm(model):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    return l1_norm

def weight_norms(model):
    norm = {name: None for name in ["input_encoder", "node_encoder", "node_mlp", "edge_mlp", "model"]}
    for key, value in model.state_dict().items():
        for name in norm.keys():
            if name in key and "weight" in key:
                norm[name] = value.norm() if norm[name] is None else norm[name] * value.norm()
    return norm


def load_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def setup_config(args):
    config = load_config(os.path.join("config","default_config.yaml"))
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    config['commit'] = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    
    return config


def init_wandb(config):
    if config["wandb"]:
        wandb.init(
            project=f"trunk-{config['model']}-0.1.0",
            config=config
        )
    else:
        wandb.init(mode="disabled", config=config)


def set_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"[WARNING] Manual seed is set to {seed}.")


def save_model(model, dataset):
    if os.path.exists("artifacts"):
        shutil.rmtree("artifacts")
    os.makedirs("artifacts")

    model_artifact = wandb.Artifact("ResidualGNN-0-debug", type="model")

    torch.save(model, os.path.join("artifacts", "model.pt"))
    model_artifact.add_file(os.path.join("artifacts", "model.pt"))

    torch.save(model.state_dict(), os.path.join("artifacts", "model_state_dict.pth"))
    model_artifact.add_file(os.path.join("artifacts", "model_state_dict.pth"))

    model_data = {
        "model_state_dict": model.state_dict(),
        "normalization_metrics": dataset.metrics,
        "config": dict(wandb.config),
    }

    torch.save(model_data, os.path.join("artifacts", "model_data.pth"))
    model_artifact.add_file(os.path.join("artifacts", "model_data.pth"))

    for file in os.listdir(os.path.join("src", "trunk_gnn")):
        if ".py" in file:
            model_artifact.add_file(os.path.join("src", "trunk_gnn", file))

    model_artifact.save()


def epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_data_loader: torch.utils.data.DataLoader,
    validation_data_loader: torch.utils.data.DataLoader,
    test_data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    gradient_clipping_max_norm: Optional[float] = None
):
    
    mask = torch.vstack((torch.eye(3)*0, torch.eye(3)*1)).to(device)

    # Training
    model.train()
    train_loss = 0
    for train_batch in train_data_loader:
        optimizer.zero_grad()
        pred = model(train_batch)
        loss = criterion(pred.x_new@mask, train_batch.x_new@mask)
        
        # Add L1 regularization
        loss += compute_l1_norm(model) * wandb.config["l1_weight"]
        
        loss.backward()

        if gradient_clipping_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clipping_max_norm
            )

        optimizer.step()
        train_loss += loss.item()
        #wandb.log({"train_batch_loss": loss.item(), "gradien_norm": torch.nn.utils.n})
    train_loss /= len(train_data_loader)

    # Validation
    model.eval()

    validation_loss = 0
    with torch.no_grad():
        for validation_batch in validation_data_loader:
            validation_loss += criterion(model(validation_batch).x_new@mask, validation_batch.x_new@mask).item()
        validation_loss /= len(validation_data_loader)

    # Test
    test_loss, zoh_loss = 0, 0
    metrics = {}
    with torch.no_grad():
        for test_batch in test_data_loader:
            pred = model(test_batch)
            for key, value in pred.metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
            
            test_loss += criterion(pred.x_new@mask, test_batch.x_new@mask).item()
            zoh_loss += criterion(test_batch.x@mask, test_batch.x_new@mask).item()
        test_loss /= len(test_data_loader)
        zoh_loss /= len(test_data_loader)

    for key, value in metrics.items():
        metrics[key] = torch.mean(torch.tensor(value)).item()
    
    wandb.log(
        {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "test_loss": test_loss,
            "zoh_test_loss": zoh_loss,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "weight_norms": weight_norms(model),
            **metrics
        }
    )

    scheduler.step(validation_loss)

