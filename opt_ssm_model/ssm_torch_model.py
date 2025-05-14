
import pickle
import torch
import numpy as np
import jax.numpy as jnp

from torch_geometric.data import Data

class SSMModel:
    def __init__(self, model_type="ssmr_orth"):
        with open(f"{model_type}.pkl", "rb") as f:
            self.model = pickle.load(f)

        self.x0 = None
        self.dt = 0.01

    def reset(self):
        self.x0 = None
        
    def eval(self):
        pass

    def init_x0(self, x0):
        z0 = x0[:3].copy()
        self.offset = z0[2]
        z0[2] -= self.offset

        z = jnp.array([z0 for _ in range(self.model.ssm.N_obs_delay + 1)]).flatten()

        self.x0 = self.model.encode(z)
        

    def __call__(self, state):
        if self.x0 is None:
            self.init_x0(state.x[-1].cpu().numpy()[:3])

        links = len(state.x)
        
        u = state.u[0].cpu().numpy()

        next_x, _ = self.model.dynamics_step(self.x0, jnp.array(u.tolist() + [self.dt]))
        self.x0 = next_x

        next_x_tensor = torch.zeros(links, 6)
        decoded = np.array(self.model.decode(next_x))
        new_z = decoded[-3:]
        new_z[2] += self.offset
        next_x_tensor[:, :3] = torch.tensor(new_z)

        return Data(x_new=next_x_tensor)
    

if __name__ == "__main__":
    # Example usage
    model = SSMModel(model_type="ssmr_orth")
    state = Data(x=torch.randn(30, 6), u=torch.randn(30,6))
    next_state = model(state).x_new
    print(next_state)