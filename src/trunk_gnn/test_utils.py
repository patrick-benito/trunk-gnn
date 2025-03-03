import numpy as np
import torch
import wandb
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from trunk_sim import visualization

def test_rollout(model: torch.nn.Module, ground_truth: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    state = ground_truth[0].clone()
    state.x_new = None # Not used in rollout

    state_list = []
    state_list_gt = []

    for i in range(len(ground_truth)):
        gt = ground_truth[i]
        state_list.append(state.x)
        state_list_gt.append(gt.x)
        
        state.t, state.u = gt.t, gt.u
        x_new = model(state).x_new
        state.x = x_new

    state_list = torch.stack(state_list)
    state_list_gt = torch.stack(state_list_gt)

    return state_list, state_list_gt

def open_loop_link_rmse(states: torch.Tensor, states_gt: torch.Tensor, links = None) -> torch.Tensor:
    if links is None:
        links = [-1]

    tip_states = states[:, links, :]
    tip_states_gt = states_gt[:, links, :]

    return torch.sqrt(torch.mean((tip_states - tip_states_gt) ** 2))

def plot_rollout(states: torch.Tensor, states_gt: torch.Tensor, links = None):
    if links is None:
        links = [-1]

    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111, projection='3d')

    states = states.detach().cpu().numpy()
    states_gt = states_gt.detach().cpu().numpy()

    for i in links:
        ax.plot(states[:, i, 0], states[:, i, 1], states[:, i, 2], label='Prediction')
        ax.plot(states_gt[:, i, 0], states_gt[:, i, 1], states_gt[:, i, 2], label='Ground Truth')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.legend()

    return fig


def open_loop_test(model: torch.nn.Module, test_data_loader: Data):
    states, gt_states = test_rollout(model, list(test_data_loader))
    rmse = open_loop_link_rmse(states, gt_states)
    fig = plot_rollout(states, gt_states)

    wandb.log({"open_loop_rmse": rmse.item(), "open_loop_rollout_fig": wandb.Image(fig)})