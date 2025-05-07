import torch
from typing import Literal
from torch_geometric.data import Data, InMemoryDataset, Dataset
import pandas as pd
from trunk_sim.data import get_column_names
import os
from torch_geometric.data import Data

def add_noise(data, noise_std=0.1):
    pass

#TODO: evaluate whether standarization or normalization is better
def normalize_data(data, metrics):
    for key, value in data.items():
        if key in metrics and value.numel() > 0:
            value = 2 * (value - metrics[key]['median']) / (metrics[key]['quantile_95'] - metrics[key]['quantile_05'])
            #value = (value-metrics[key]['mean'])/metrics[key]['std']
            data[key] = value
    
    return data

def denormalize_data(data, metrics):
    for key, value in data.items():
        if key in metrics and value.numel() > 0:
            value = value*(metrics[key]['quantile_95']-metrics[key]['quantile_05'])/2 + metrics[key]['median']
            data[key] = value
    
    return data


class TrunkGraphDataset(InMemoryDataset):
    def __init__(self, root,  normalization_strategy: Literal['none', 'normalize', 'inverse'] = 'none', add_noise = False, metrics = None, device = None, link_step = 1):
        
        self.link_step = link_step
        force_reload = False

        try:
            if self.link_step != int(torch.load(os.path.join(root, 'processed', 'link_step.pt'))):
                force_reload = True
                print("Link step changed, reloading dataset.")
        except FileNotFoundError:
            pass


        super().__init__(root, transform=self.transform, force_reload=force_reload) # or use NormalizeFeatures
        self.root = root
        
        self.load(self.processed_paths[0])
        self.num_links = torch.load(self.processed_paths[1])
        if device:
            self.to(device)
            
        self.normalization_strategy = normalization_strategy
        self.metrics = metrics if metrics else self.compute_metrics()
        self.add_noise = add_noise
        
        print(f"Loaded {root} dataset containing {len(self)} samples.")

    @property
    def raw_file_names(self):
        return ['data.csv']
    
    @property
    def processed_file_names(self):
        return ['processed_data.pt', 'num_links.pt', 'link_step.pt']
    
    def compute_metrics(self):
        metrics = {}
        for key, value in self._data.items('x','u','x_new'):
            if value.numel() > 0:
                device = value.device

                metrics[key] = {
                    "median": value.median(0).values.to(device),
                    "quantile_95": torch.quantile(value,0.95,dim=0).to(device),
                    "quantile_05": torch.quantile(value,0.05,dim=0).to(device),
                    "max": value.max(0).values.to(device),
                    "min": value.min(0).values.to(device),
                    "std": value.std(0).to(device),
                    "mean": value.mean(0).to(device)
                }
        
        #metrics['y'] = metrics['x'] # apply metrics to target y

        return metrics

    def transform(self,data):
        if self.add_noise:
            data = self.add_noise(data)
        if self.normalization_strategy == "normalize":
            return normalize_data(data, self.metrics)
        elif self.normalization_strategy == "denormalize":
            return denormalize_data(data,self.metrics)
        elif self.normalization_strategy == "none":
            return data
        else:
            raise ValueError("Normalization strategy {self.normalization_strategy} was not recognized.")

    def process(self):
        # The process function is only executed if processed_files are missing.
        data_list = []
        for raw_path in self.raw_paths:
            self.dataframe = pd.read_csv(raw_path)
            self.time_col = "t"

            #TODO: Currently infering from dataframe, should be passed as argument or metadata file.
            self.num_segments = len([key for key in self.dataframe.keys() if key.startswith("ux")])

            num_links_mujoco = len([key for key in self.dataframe.keys() if key.startswith("x") and not key.endswith("_new")])

            if isinstance(self.link_step, int):
                self.links = list(range(num_links_mujoco, 0, -self.link_step))[::-1] # Ensures that endeffector is always included
                print("Links:", self.links)
            else:
                raise ValueError("Link step should be an integer.")
            
            self.num_links = len(self.links)
                      
            self.state_cols, self.state_new_cols, self.control_cols = get_column_names(
                self.num_segments, "pos_vel", self.links
            )

            d = 1 # d-local connections (currently in form of a chain)
            edge_index = torch.tensor([[i, j] for i in range(self.num_links) for j in range(self.num_links) if 0 < abs(i-j) < d+1]).T

            for idx in range(len(self.dataframe)):
                ids = self.links
                t = self.dataframe.iloc[idx][self.time_col]
                states = self.dataframe.iloc[idx][self.state_cols].values
                controls = self.dataframe.iloc[idx][self.control_cols].values
                next_states = self.dataframe.iloc[idx][self.state_new_cols].values

                # Convert to PyTorch tensors
                t = torch.tensor(t, dtype=torch.float32)
                x = torch.tensor(states, dtype=torch.float32).reshape(self.num_links, -1)
                u = torch.tensor(controls, dtype=torch.float32).repeat(self.num_links, 1)
                x_new = torch.tensor(next_states, dtype=torch.float32).reshape(self.num_links, -1)
                ids = torch.tensor(ids, dtype=torch.float32).reshape(self.num_links, -1)

                """ if idx == 0:
                    # This is only valid in sim
                    if torch.allclose(x[:, 0], torch.zeros_like(x[:, 0])) and torch.allclose(x[:, 1], torch.zeros_like(x[:, 1])):
                        print("Resting state of links is as expected.")
                        assert torch.allclose(x[:, 2], torch.tensor([-0.0106666666666666 * i for i in range(1, self.num_links+1)])), f"Resting state of links is not as expected."
                """
                data_list.append(Data(t=t, x=x, u=u, x_new=x_new, edge_index=edge_index, ids=ids))

        self.save(data_list, self.processed_paths[0])
        torch.save(self.num_links, self.processed_paths[1])
        torch.save(self.link_step, self.processed_paths[2])
