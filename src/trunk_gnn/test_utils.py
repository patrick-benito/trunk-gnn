import numpy as np
import torch
import wandb
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from trunk_sim import visualization
import os
from torch_geometric.loader import DataLoader
from trunk_gnn.data import TrunkGraphDataset
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_rollout(model: torch.nn.Module, ground_truth: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    start_time = time.time()
    start_index = 0

    model.eval()
    state = ground_truth[start_index].clone()
    state.x_new = None # Not used in rollout

    state_list = []
    state_list_gt = []

    for i in range(start_index,len(ground_truth)):
        gt = ground_truth[i]
        state_list.append(state.x)
        state_list_gt.append(gt.x)
        
        state.t, state.u = gt.t, gt.u
        x_new = model(state).x_new
        state.x = x_new

    state_list = torch.stack(state_list)
    state_list_gt = torch.stack(state_list_gt)

    delta_time = time.time() - start_time

    return state_list, state_list_gt, delta_time

def open_loop_link_rmse(states: torch.Tensor, states_gt: torch.Tensor, links = None) -> torch.Tensor:
    if links is None:
        links = [-1]

    tip_states = states[:, links, :]
    tip_states_gt = states_gt[:, links, :]

    return torch.sqrt(torch.mean((tip_states - tip_states_gt) ** 2))

def plot_rollout(states: torch.Tensor, states_gt: torch.Tensor, links = None):
    if links is None:
        links = [-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    states = states.detach().cpu().numpy()
    states_gt = states_gt.detach().cpu().numpy()

    for i in links:
        ax.plot(states[:, i, 0], states[:, i, 1], states[:, i, 2], label=f'Prediction {i}')
        ax.plot(states_gt[:, i, 0], states_gt[:, i, 1], states_gt[:, i, 2], label=f'Ground Truth {i}')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.legend()

    return fig

def plot_velocities(states: torch.Tensor, states_gt: torch.Tensor, links=None):
    if links is None:
        links = [-1]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    states = states.detach().cpu().numpy()
    states_gt = states_gt.detach().cpu().numpy()

    for i in links:
        ax[0].plot(states[:, i, 3], label=f'Prediction $v_x{i}$')
        ax[0].plot(states_gt[:, i, 3], label=f'Ground Truth $v_x{i}$')
        ax[1].plot(states[:, i, 4], label=f'Prediction $v_y{i}$')
        ax[1].plot(states_gt[:, i, 4], label=f'Ground Truth $v_y{i}$')
        ax[2].plot(states[:, i, 5], label=f'Prediction $v_z{i}$')
        ax[2].plot(states_gt[:, i, 5], label=f'Ground Truth $v_z{i}$')

    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Velocity $x$')
    ax[0].legend()
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity $y$')
    ax[1].legend()
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Velocity $z$')
    ax[2].legend()

    return fig


def open_loop_test(model: torch.nn.Module, test_data_loader: Data, additonal_info: str = "", save_figures = False) -> torch.Tensor:
    states, gt_states, delta_time = test_rollout(model, list(test_data_loader))
    rmse = open_loop_link_rmse(states, gt_states)
    fig_positions = plot_rollout(states, gt_states, links=[0, 10, 20, -1])
    fig_velocities = plot_velocities(states, gt_states, links=[0, 10, 20, -1])
    
    if save_figures:
        fig_positions.savefig(f"figures/open_loop_rollout_fig_positions_{additonal_info}.png")
        fig_velocities.savefig(f"figures/open_loop_rollout_fig_velocities_{additonal_info}.png")

    wandb.log({f"rollout_time_{additonal_info}":delta_time, f"open_loop_rmse_{additonal_info}": rmse.item(), f"open_loop_rollout_fig_positions_{additonal_info}": wandb.Image(fig_positions), f"open_loop_rollout_fig_velocities_{additonal_info}": wandb.Image(fig_velocities)}, commit=False)

    plt.close(fig_positions)
    plt.close(fig_velocities)
    return rmse


def open_loop_test_all(model: torch.nn.Module, test_data_folder: str, save_figures = False) -> torch.Tensor:
    avg_open_loop_rmse = 0
    if save_figures:
        os.makedirs("figures", exist_ok=True)

    folders = os.listdir(test_data_folder)
    if "processed" in folders:
        folders = ["."]

    for folder in folders:
        test_data_loader = DataLoader(TrunkGraphDataset(os.path.join(test_data_folder, folder), device=device))
        avg_open_loop_rmse += open_loop_test(model, test_data_loader, additonal_info=folder, save_figures=save_figures)

    avg_open_loop_rmse /= len(os.listdir(test_data_folder))
    
    wandb.log({"avg_open_loop_rmse": avg_open_loop_rmse.item()}, commit=False)
    return avg_open_loop_rmse