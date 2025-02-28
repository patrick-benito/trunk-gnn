import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data


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

    def initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class GNNBlock(MessagePassing):
    def __init__(
        self, node_channels_in, edge_channels_in, node_channels_out, edge_channels_out
    ):
        super().__init__(aggr="sum", flow="source_to_target")
        # self.edge_mlp = MLP(edge_channels*0 + 1*node_channels, 50, hidden_features=150, num_hidden_layers=3) # e' <- f_e(e, x_i, x_j)
        # self.node_mlp = MLP(node_channels + 50, 2, hidden_features=100, num_hidden_layers=1) # x' <- f_v(x, e)
        self.edge_mlp = MLP(
            edge_channels_in,
            edge_channels_out,
            num_hidden_layers=4,
            hidden_features=150,
        )
        self.node_mlp = MLP(
            node_channels_in + edge_channels_out,
            node_channels_out,
            num_hidden_layers=1,
            hidden_features=100,
        )

    def forward(self, x, edge_index, edge_attr):
        # edge_index, _ = add_self_loops(edge_index) #TODO: add self loops

        # Update edge features
        marsh_edge_attr = self.update_edges(x, edge_index, edge_attr)

        # Call message and update consecutively
        x = self.propagate(edge_index, x=x, marsh_edge_attr=marsh_edge_attr)

        return x, edge_attr

    def update_edges(self, x, edge_index, edge_attr):
        # Marshalling TODO: stack with edge_attr
        sender, receiver = edge_index
        diff = x[receiver] - x[sender]
        m = torch.cat([x[receiver], x[sender], diff], dim=1)
        return self.edge_mlp(m)

    def message(self, marsh_edge_attr):
        return marsh_edge_attr

    def update(self, aggr_out, x):
        x = self.node_mlp(torch.cat([x[:, :], aggr_out], dim=1))
        return x


class ResidualGNN(nn.Module):
    def __init__(self, node_channels, edge_channels, num_blocks=1):
        super(ResidualGNN, self).__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.layers.append(
                GNNBlock(
                    node_channels_in=5,
                    node_channels_out=2,
                    edge_channels_in=3 * 5,
                    edge_channels_out=50,
                )
            )

        # self.node_encode = MLP(5, 50, layernorm=False)
        # self.node_decode = MLP(2, 2, layernorm=False, num_hidden_layers=10)

    def forward(self, data: Data):
        # x = self.node_encode(data.x)
        edge_attr = data.edge_attr if data.edge_attr is not None else None

        for layer in self.layers:
            dx, dedge_attr = layer(data.x, data.edge_index, edge_attr)

            # x += dx
            # x[:,3:5] = dx
            # x[:,1:3] += dx * 0.001

        # dx = self.node_decode(dx)

        return Data(x=dx)
