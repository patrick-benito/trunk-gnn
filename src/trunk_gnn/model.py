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
        batchnorm=False,
    ):
        super(MLP, self).__init__()

        layers = []
        if num_hidden_layers == 0:
            layers.append(nn.Linear(in_features, out_features))
        else:
            layers.append(nn.Linear(in_features, hidden_features))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_features))
            layers.append(nn.ReLU())
            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(hidden_features, hidden_features))
                if batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_features))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_features, out_features))

        if layernorm:
            layers.append(nn.LayerNorm(out_features))

        self.model = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)


class GNNBlock(MessagePassing):
    def __init__(
        self, node_channels_in, edge_channels_in, node_channels_out, edge_channels_out
    ):
        node_encoder_out = edge_channels_out if wandb.config["encode_nodes"] else node_channels_in
        input_encoder_out = edge_channels_out if wandb.config["encode_inputs"] else 6
        if not wandb.config["use_inputs"]:
            input_encoder_out = 0

        super().__init__(aggr="sum", flow="source_to_target")

        if wandb.config["encode_nodes"]:
            self.node_encoder = MLP(
                node_channels_in,
                node_encoder_out,
                num_hidden_layers=4,
                hidden_features=50,
            )
            
        if wandb.config["encode_inputs"]:
            self.input_encoder = MLP(
                6,
                input_encoder_out,
                num_hidden_layers=4,
                hidden_features=50,
            )
        

        self.edge_mlp = MLP(
            edge_channels_in,
            edge_channels_out,
            num_hidden_layers=wandb.config["edge_num_hidden_layers"],
            hidden_features=wandb.config["edge_hidden_features"],
            batchnorm=wandb.config["batch_norm"],
            layernorm=wandb.config["layer_norm"],

        ) # e' <- f_e(e, x_i, x_j)

        self.node_mlp = MLP(
            node_encoder_out + edge_channels_out + input_encoder_out,
            node_channels_out,
            num_hidden_layers=wandb.config["node_num_hidden_layers"],
            hidden_features=wandb.config["node_hidden_features"],
            batchnorm=wandb.config["batch_norm"],
            layernorm=wandb.config["layer_norm"],
        ) # x' <- f_v(x, e)

    def forward(self, x, edge_index, edge_attr=None, u=None):
        # Update edge features
        marsh_edge_attr = self.update_edges(x, edge_index, edge_attr)
        
        # Call message and update consecutively
        if wandb.config["encode_inputs"]:
            u = self.input_encoder(u)

        if wandb.config["encode_nodes"]:
            x = self.node_encoder(x)

        x, metrics = self.propagate(edge_index, x=x, marsh_edge_attr=marsh_edge_attr, u=u)

        return x, edge_attr, metrics

    def update_edges(self, x, edge_index, edge_attr=None):
        sender, receiver = edge_index
        if wandb.config["use_edge_mlp_diff_features_only"]:
            edge_large = x[receiver] - x[sender]
        else:
            edge_large = torch.cat([x[sender], x[receiver]], dim=1)

        return self.edge_mlp(edge_large)

    def message(self, marsh_edge_attr):
        return marsh_edge_attr

    def update(self, aggr_out, x, u):
        metrics = {}
        if not self.training and wandb.config["log_metrics"]:
            metrics = {
                "x_norm": x.norm(dim=1).mean(),
                "u_norm": u.norm(dim=1).mean(),
                "aggr_out_norm": aggr_out.norm(dim=1).mean()
            }

        if wandb.config["use_velocity_only"]:
            x[:,:3] = 0

        if not wandb.config["use_edge_mlp"]:
            aggr_out = torch.zeros_like(aggr_out)
        
        if wandb.config["use_inputs"]:
            x = torch.hstack([x, u])

        x_large = torch.cat([x, aggr_out], dim=1)
        
        return self.node_mlp(x_large), metrics


## GNN model ##

class TrunkGNN(nn.Module):
    def __init__(self, num_blocks=1):
        super(TrunkGNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList()
        self.num_links = 30
        self.num_blocks = 1
        self.dt = 0.01
        self.link_delta_z_pos = -0.0106666666666666 # TODO: this is bad for edge mlp
        # compute the resting state of the links as [0,0,link_delta_z_pos * i,0,0,0] for i in (1,2,3,...,30). We incude 0 so that index matches the link number
        self.x_rest = torch.kron(torch.tensor([[0, 0, self.link_delta_z_pos, 0, 0, 0]]), torch.tensor(range(0, self.num_links+1)).reshape(-1,1)).to(self.device)

        self.alpha_mlp = MLP(1, 1, num_hidden_layers=1, hidden_features=5)
        self.alpha_mlp_inv = MLP(1, 1, num_hidden_layers=1, hidden_features=5)

        node_channels_in = 6

        if wandb.config["use_ids"]:
            node_channels_in += 1

        for _ in range(num_blocks):
            self.layers.append(
                GNNBlock(
                    node_channels_in=node_channels_in,
                    node_channels_out=3,
                    edge_channels_in=node_channels_in if wandb.config['use_edge_mlp_diff_features_only'] else node_channels_in * 2,
                    edge_channels_out=wandb.config["edge_channels_out"],
                )
            )

    def forward(self, _data: Data):
        data = _data.clone()
        
        layer = self.layers[0]

        x = data.x
        ids = data.ids
        ids_int = ids.flatten().int()
        x_bar = x

        if wandb.config["use_resting_state"]:
            x_bar = x - self.x_rest[ids_int]
        
        if wandb.config["use_alpha"]:
            x_bar = x_bar * self.alpha_mlp(ids)
            
        if wandb.config["use_ids"]:
            x_bar = torch.hstack([x_bar, ids/30])

        dv, _, metrics = layer(x_bar, data.edge_index, data.edge_attr, data.u)
        
        if wandb.config["use_alpha"]:
            dv = dv * self.alpha_mlp_inv(ids)
  
        v_new = x[:,3:] + dv                          # Update velocity
        x_new = x[:,:3] + v_new * self.dt             # Update position

        full_x_new = torch.cat([x_new, v_new], dim=1)
        
        return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, t=data.t, u=data.u, x_new=full_x_new, metrics=metrics)


## MLP model as a baseline ##
class TrunkMLP(nn.Module):
    def __init__(self):
        super(TrunkMLP, self).__init__()
    
        self.num_links = int(30/wandb.config["link_step"])
        self.in_features = 6*self.num_links
        if wandb.config["use_inputs"]:
            self.in_features += 6
        self.out_features = 3*self.num_links
        self.model = MLP(self.in_features, self.out_features, num_hidden_layers=4, hidden_features=150)
        self.dt = 0.01

    def forward(self, _data: Data):
        data = _data.clone()
        #TODO: Only handles unshuffled data
        
        x = data.x
        u = data.u

        if wandb.config["use_inputs"]:
            vec = torch.hstack([x.view(-1,6*self.num_links), u.view(-1, self.num_links, 6)[:,0,:]])
        else:
            vec = x.view(-1,self.in_features)

        dv = self.model(vec)
        dv = dv.view(-1, 3)
        
        v_new = x[:,3:] + dv
        x_new = x[:,:3] + v_new * self.dt   # Update position

        full_x_new = torch.cat([x_new, v_new], dim=1)

        return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, t=data.t, u=data.u, x_new=full_x_new, metrics={})
