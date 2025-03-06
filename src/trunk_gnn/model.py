import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import wandb

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

from trunk_gnn.masks import velocity_mask, position_mask

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers=2,
        hidden_features=128,
        layernorm=False,
    ):
        super(MLP, self).__init__()

        if num_hidden_layers == 0:
            layers = [nn.Linear(in_features, out_features)]
        else:
            layers = [nn.Linear(in_features, hidden_features), nn.ReLU()]
            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(hidden_features, hidden_features))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_features, out_features))

        if layernorm:
            layers.append(torch.nn.LayerNorm(out_features))

        self.model = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)


class GNNBlock(MessagePassing):
    def __init__(
        self, node_channels_in, edge_channels_in, node_channels_out, edge_channels_out
    ):
        super().__init__(aggr="sum", flow="source_to_target")
        self.edge_mlp = MLP(
            edge_channels_in,
            edge_channels_out,
            num_hidden_layers=wandb.config.get("edge_num_hidden_layers", 4),
            hidden_features=wandb.config.get("edge_hidden_features", 150),
        ) # e' <- f_e(e, x_i, x_j)

        self.node_mlp = MLP(
            node_channels_in + edge_channels_out,
            node_channels_out,
            num_hidden_layers=wandb.config.get("node_num_hidden_layers", 1),
            hidden_features=wandb.config.get("node_hidden_features", 100),
        ) # x' <- f_v(x, e)

    def forward(self, x, edge_index, edge_attr=None):
        # Update edge features
        marsh_edge_attr = self.update_edges(x, edge_index, edge_attr)

        # Call message and update consecutively
        x = self.propagate(edge_index, x=x, marsh_edge_attr=marsh_edge_attr)

        return x, edge_attr

    def update_edges(self, x, edge_index, edge_attr=None):
        sender, receiver = edge_index
        diff = x[receiver] - x[sender]

        return self.edge_mlp(torch.cat([diff], dim=1))

    def message(self, marsh_edge_attr):
        return marsh_edge_attr

    def update(self, aggr_out, x):

        x = self.node_mlp(torch.cat([x, aggr_out], dim=1))
        return x


## GNN model ##

class TrunkGNN(nn.Module):
    def __init__(self, num_links, num_blocks=1):
        super(TrunkGNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList()
        self.num_links = num_links
        self.num_blocks = 1
        self.dt = 0.01

        for _ in range(num_blocks):
            self.layers.append(
                GNNBlock(
                    node_channels_in=6,
                    node_channels_out=3,
                    edge_channels_in=6,
                    edge_channels_out=wandb.config.get("edge_channels_out", 50),
                )
            )

    def forward(self, _data: Data):
        data = _data.clone()
        for layer in self.layers:
            x = data.x
            dv, _ = layer(x, data.edge_index, data.edge_attr)
            
            v_new = x[:,3:] + dv                          # Update velocity
            x_new = x[:,:3] + v_new * self.dt              # Update position

            full_x_new = torch.cat([x_new, v_new], dim=1)
            
        return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, t=data.t, u=data.u, x_new=full_x_new)


## MLP model as a baseline ##

class TrunkMLP(nn.Module):
    def __init__(self, num_links):
        super(TrunkMLP, self).__init__()
    
        self.num_links = num_links
        self.in_features = 3*num_links
        self.out_features = 3*num_links
        self.model = MLP(self.in_features, self.out_features, num_hidden_layers=4, hidden_features=150)
        self.dt = 0.01

    def forward(self, _data: Data):
        data = _data.clone()
        #TODO: Only handles unshuffled data
        
        x = data.x
        vel = x@velocity_mask
        dv = self.model(vel.view(-1, self.in_features))
        dv = dv.view(-1, 3)

        v_new = x[:,3:] + dv
        x_new = x[:,:3] + v_new * self.dt   # Update position

        full_x_new = torch.cat([x_new, v_new], dim=1)

        return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, t=data.t, u=data.u, x_new=full_x_new)
