
import pickle
import torch
import numpy as np
import jax.numpy as jnp
import os
import sys

from torch_geometric.data import Data

sys.path.append(os.path.join(os.path.dirname(__file__), "opt_ssm"))

class SSMModel:
    def __init__(self, model_type, folder_path = None):
        folder_path = os.path.join(os.path.dirname(__file__)) if folder_path is None else folder_path

        model_path = os.path.join(folder_path, f"{model_type}.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.x0 = None
        self.dt = 0.01

        z_offset_path = os.path.join(folder_path, "z_offset.pkl")
        with open(z_offset_path, "rb") as f:
            self.z_offset = pickle.load(f)

    def reset(self):
        self.x0 = None
        
    def eval(self):
        pass

    def init_x0(self, states: list):
        assert len(states) == 4, "The number of delay embeddings must be 4."

        state_array = np.array([
            state.x[-1].cpu().numpy().copy()[:3] for state in states
        ])

        z0 = state_array[::-1] #most recent first
        z0[:,2] -= self.z_offset

        print(z0.shape)
        self.x0 = self.model.encode(z0.flatten())

    def __call__(self, state):
        if self.x0 is None:
            raise ValueError("Initial state x0 is not set. Call init_x0() first.")

        links = len(state.x)
        
        u = state.u[0].cpu().numpy()

        next_x, _ = self.model.dynamics_step(self.x0, jnp.array(u.tolist() + [self.dt]))
        self.x0 = next_x

        next_x_tensor = torch.full((links, 6), float('nan'))

        decoded = np.array(self.model.decode(next_x))
        new_z = decoded[:3].copy()
        new_z[2] += self.z_offset
        next_x_tensor[-1, :3] = torch.tensor(new_z)

        return Data(x_new=next_x_tensor)
    

if __name__ == "__main__":
    # Example usage
    model = SSMModel(model_type="ssmr_orth")
    state = Data(x=torch.randn(30, 6), u=torch.randn(30,6))
    model.init_x0([state for _ in range(4)])
    for i in range(2):
        print(model(state).x_new)