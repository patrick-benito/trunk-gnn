import numpy as np
import torch
import wandb
from torch_geometric.data import Data
import time
from typing import List
import os

from torch_geometric.loader import DataLoader
from trunk_gnn.data import TrunkGraphDataset
from trunk_gnn.plotting import plt

from trunk_gnn.train_utils import setup_config

map_link = lambda link, n_links: min(max(round(link/30*n_links) - 1, 0), n_links-1)
map_link_mujoco = lambda link, n_links: round((map_link(link,n_links)+1) / n_links * 30)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_sets_from_folder(test_data_folder: str, link_step = 1) -> List[TrunkGraphDataset]:
    datasets = []

    folders = [f for f in os.listdir(test_data_folder) if os.path.isdir(os.path.join(test_data_folder, f))]

    if "raw" in folders:
        folders = ["."]

    folders = sorted(folders, key=lambda x: int(x.split("/")[-1]))

    for folder in folders:
        if "raw" in os.listdir(os.path.join(test_data_folder, folder)):
            datasets.append(DataLoader(TrunkGraphDataset(os.path.join(test_data_folder, folder), device=device, link_step=link_step)))

    return datasets

def test_rollout(model: torch.nn.Module, ground_truth: list[Data], start_index = 0) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        start_time = time.time()

        model.eval()
        state = ground_truth[start_index].clone()
        state.x_new = None # Not used in rollout

        state_list = []
        state_list_gt = []

        for i in range(start_index,len(ground_truth)):
            gt = ground_truth[i]
            state_list.append(state.x.detach())
            state_list_gt.append(gt.x)
            
            state.t, state.u = gt.t, gt.u
            x_new = model(state).x_new
            state.x = x_new

        state_list = torch.stack(state_list)
        state_list_gt = torch.stack(state_list_gt)

        delta_time = time.time() - start_time

        return state_list, state_list_gt, delta_time

def open_loop_link_rmse(states: torch.Tensor, states_gt: torch.Tensor, links) -> torch.Tensor:
    N = states.shape[1]
    mapped = [map_link(link, N) for link in links]
    target_states = states[:, mapped, :3]
    target_states_gt = states_gt[:, mapped, :3]

    return torch.sqrt(torch.mean((target_states - target_states_gt) ** 2))

def open_loop_link_se(states: torch.Tensor, states_gt: torch.Tensor, links) -> torch.Tensor:
    N = states.shape[1]
    mapped = [map_link(link, N) for link in links]
    target_states = states[:, mapped, :3]
    target_states_gt = states_gt[:, mapped, :3]
    return torch.sum(torch.mean((target_states - target_states_gt) ** 2, axis=2), axis=0)


def plot_rollout(states: torch.Tensor, states_gt: torch.Tensor, links):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')

    states = states.detach().cpu().numpy()
    states_gt = states_gt.detach().cpu().numpy()
    N = states.shape[1]

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, color in zip(reversed(links), color_cycle):
        ax.plot(states_gt[:, map_link(i,N), 0], states_gt[:, map_link(i,N), 1], states_gt[:, map_link(i,N), 2], label=f'Ground Truth {map_link_mujoco(i,N)}', linestyle='--', color=color, linewidth=0.5)
        ax.plot(states[:, map_link(i,N), 0], states[:, map_link(i,N), 1], states[:, map_link(i,N), 2], label=f'Prediction {map_link_mujoco(i,N)}', color=color)
    
    ax.title.set_text('Position Rollout') 
    ax.grid(False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.legend()

    return fig

def plot_positions_1d(states: torch.Tensor, states_gt: torch.Tensor, links):
    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    states = states.detach().cpu().numpy()
    states_gt = states_gt.detach().cpu().numpy()
    N = states.shape[1]

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, color in zip(reversed(links), color_cycle):
        ax[0].plot(states[:, map_link(i,N), 0], label=f'Prediction $x^{{{map_link_mujoco(i,N)}}}$')
        ax[0].plot(states_gt[:, map_link(i,N), 0], label=f'Ground Truth $x^{{{map_link_mujoco(i,N)}}}$', linestyle='--', color=color)
        ax[1].plot(states[:, map_link(i,N), 1], label=f'Prediction $y^{{{map_link_mujoco(i,N)}}}$', color=color)
        ax[1].plot(states_gt[:, map_link(i,N), 1], label=f'Ground Truth $y^{{{map_link_mujoco(i,N)}}}$', linestyle='--', color=color)
        ax[2].plot(states[:, map_link(i,N), 2], label=f'Prediction $z^{{{map_link_mujoco(i,N)}}}$', color=color)
        ax[2].plot(states_gt[:, map_link(i,N), 2], label=f'Ground Truth $z^{{{map_link_mujoco(i,N)}}}$', linestyle='--', color=color)
    
    ax[0].set_title('$x$-Position Rollout')
    ax[0].set_xlabel('Time Step $k$')
    ax[0].set_ylabel('Position $x$')
    ax[1].set_title('$y$-Position Rollout')
    ax[1].set_xlabel('Time Step $k$')
    ax[1].set_ylabel('Position $y$')
    ax[2].set_title('$z$-Position Rollout')
    ax[2].set_xlabel('Time Step $k$')
    ax[2].set_ylabel('Position $z$')
    fig.subplots_adjust(bottom=0.25)

    # Place legends at the bottom
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    return fig

def plot_velocities(states: torch.Tensor, states_gt: torch.Tensor, links):
    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    states = states.detach().cpu().numpy()
    states_gt = states_gt.detach().cpu().numpy()
    N = states.shape[1]

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, color in zip(reversed(links), color_cycle):
        ax[0].plot(states[:, map_link(i,N), 3], label=f'Prediction $v_x^{{{map_link_mujoco(i,N)}}}$')
        ax[0].plot(states_gt[:, map_link(i,N), 3], label=f'Ground Truth $v_x^{{{map_link_mujoco(i,N)}}}$', linestyle='--', color=color)
        ax[1].plot(states[:, map_link(i,N), 4], label=f'Prediction $v_y^{{{map_link_mujoco(i,N)}}}$', color=color)
        ax[1].plot(states_gt[:, map_link(i,N), 4], label=f'Ground Truth $v_y^{{{map_link_mujoco(i,N)}}}$', linestyle='--', color=color)
        ax[2].plot(states[:, map_link(i,N), 5], label=f'Prediction $v_z^{{{map_link_mujoco(i,N)}}}$', color=color)
        ax[2].plot(states_gt[:, map_link(i,N), 5], label=f'Ground Truth $v_z^{{{map_link_mujoco(i,N)}}}$', linestyle='--', color=color)
    
    ax[0].set_title('$x$-Velocity Rollout')
    ax[0].set_xlabel('Time Step $k$')
    ax[0].set_ylabel('Velocity $v_x$')
    ax[1].set_title('$y$-Velocity Rollout')
    ax[1].set_xlabel('Time Step $k$')
    ax[1].set_ylabel('Velocity $v_y$')
    ax[2].set_title('$z$-Velocity Rollout')
    ax[2].set_xlabel('Time Step $k$')
    ax[2].set_ylabel('Velocity $v_z$')
    fig.subplots_adjust(bottom=0.25)

    # Place legends at the bottom
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    return fig


def open_loop_test(model: torch.nn.Module, test_data_loader: Data, additonal_info: str = "", save_figures = False) -> torch.Tensor:
    states, gt_states, delta_time = test_rollout(model, list(test_data_loader), start_index=0)
    rmse = open_loop_link_rmse(states, gt_states, links = [30])
    se = open_loop_link_se(states, gt_states, links = [30])

    fig_positions = plot_rollout(states, gt_states, links=[1, 10, 20, 30])
    fig_velocities = plot_velocities(states, gt_states, links=[1, 10, 20, 30])
    fig_positions_1d = plot_positions_1d(states, gt_states, links=[1, 10, 20, 30])
    
    if save_figures:
        fig_positions.savefig(f"figures/open_loop_rollout_fig_positions_{additonal_info}.svg")
        fig_positions.savefig(f"figures/open_loop_rollout_fig_positions_{additonal_info}.pdf")
        fig_velocities.savefig(f"figures/open_loop_rollout_fig_velocities_{additonal_info}.svg")
        fig_velocities.savefig(f"figures/open_loop_rollout_fig_velocities_{additonal_info}.pdf")

    wandb.log({f"rollout_time_{additonal_info}":delta_time, f"tip_open_loop_rmse_{additonal_info}": rmse.item(), f"tip_open_loop_se_{additonal_info}": se.item()}, commit=False)
    wandb.log({f"open_loop_rollout_fig_positions_{additonal_info}": wandb.Image(fig_positions), f"open_loop_rollout_fig_velocities_{additonal_info}": wandb.Image(fig_velocities)}, commit=False)
    wandb.log({f"open_loop_rollout_fig_positions_1d_{additonal_info}": wandb.Image(fig_positions_1d)}, commit=False)

    plt.close('all')
    return rmse


def open_loop_test_all(model: torch.nn.Module, test_datasets: List[TrunkGraphDataset], save_figures = False) -> torch.Tensor:
    assert len(test_datasets) > 0, "No test datasets provided"

    avg_open_loop_rmse = 0
    if save_figures:
        os.makedirs("figures", exist_ok=True)

    for index, test_data_loader in enumerate(test_datasets):
        avg_open_loop_rmse += open_loop_test(model, test_data_loader, additonal_info=index, save_figures=save_figures)

    avg_open_loop_rmse /= len(test_datasets)
    
    wandb.log({"avg_tip_open_loop_rmse": avg_open_loop_rmse.item()}, commit=False)
    return avg_open_loop_rmse


def download_artifacts(artifact_name):
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    artifact.download(root="artifacts")


def load_model(model_type, artifacts_folder = "./artifacts/", artifact_name = None):
    if artifact_name:
        download_artifacts(artifact_name)

    model_data = torch.load(os.path.join(artifacts_folder, "model_data.pth"))
    wandb.init(config=setup_config(model_data['config']))
    print(f"Initialized wandb with config: {wandb.config}")

    model = model_type()
    model.load_state_dict(torch.load(os.path.join(artifacts_folder, "model_state_dict.pth")))
    model.to(device)

    return model