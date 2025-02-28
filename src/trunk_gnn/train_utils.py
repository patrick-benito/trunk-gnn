import wandb
import torch
import subprocess
import os

def init_wandb(args):
    if args.wandb:
        wandb.init(
            project="ResidualGNN-test-1",
            config={
                "learning_rate": args.learning_rate,
                "data_set_folder": args.data_set_folder,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "args": args,
                "notes": args.notes,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"])
                .strip()
                .decode("utf-8"),
            },
        )
    else:
        wandb.init(mode="disabled")


def set_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"[WARNING] Manual seed is set to {seed}.")


def epoch(
    model,
    optimizer,
    criterion,
    scheduler,
    train_data_loader,
    validation_data_loader,
    test_data_loader,
    device,
    gradient_clipping_max_norm=None,
):
    model.train()
    train_loss = 0
    for train_batch in train_data_loader:
        train_batch = train_batch.to(device)
        optimizer.zero_grad()
        output = model(train_batch)
        loss = criterion(output.x, train_batch.y)

        loss.backward()

        # Print average and maximum gradient norm
        if gradient_clipping_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clipping_max_norm
            )

        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_data_loader)

    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for validation_batch in validation_data_loader:
            validation_batch = validation_batch.to(device)
            loss = criterion(model(validation_batch).x, validation_batch.y)
            validation_loss += loss.item()
        validation_loss /= len(validation_data_loader)

    test_loss = 0
    zoh_loss = 0
    with torch.no_grad():
        for test_batch in test_data_loader:
            test_batch = test_batch.to(device)
            loss = criterion(model(test_batch).x, test_batch.y)
            test_loss += loss.item()
            loss = criterion(test_batch.x[:, 3:5], test_batch.y)
            zoh_loss += loss.item()
        test_loss /= len(test_data_loader)
        zoh_loss /= len(test_data_loader)

    wandb.log(
        {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "test_loss": test_loss,
            "zoh_test_loss": zoh_loss,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
    )

    scheduler.step(validation_loss)


def save_model(model, data_set):
    model_artifact = wandb.Artifact("ResidualGNN-0-debug", type="model")

    torch.save(model, "model.pt")
    model_artifact.add_file("model.pt")

    torch.save(model.state_dict(), "model.pth")
    model_artifact.add_file("model.pth")

    model_data = {
        "model_state_dict": model.state_dict(),
        "node_channels": data_set.node_channels,
        "edge_channels": data_set.edge_channels,
        "normalization_metrics": data_set.metrics,
    }

    torch.save(model_data, "model_data.pth")

    for file in os.listdir("src/algos"):
        if ".py" in file:
            model_artifact.add_file(f"src/algos/{file}")

    model_artifact.save()
