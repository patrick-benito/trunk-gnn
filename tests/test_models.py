import torch
from torch_geometric.data import Data
import pytest

from src.algos.gnn import ResidualGNN


def generate_test_data():
    # Define synthetic data
    num_nodes = 4
    node_channels = 3
    edge_channels = 2

    x = torch.randn((num_nodes, node_channels))
    edge_index = torch.tensor([[0, 2, 2, 3], [1, 1, 3, 0]], dtype=torch.long)

    num_edges = edge_index.size(1)
    edge_attr = torch.randn((num_edges, edge_channels))

    return x, edge_index, edge_attr


def test_residual_gnn_update():
    x, edge_index, edge_attr = generate_test_data()
    x_prev, edge_attr_prev = x.clone(), edge_attr.clone()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    model = ResidualGNN(
        node_channels=x.size(1), edge_channels=edge_attr.size(1), num_blocks=1
    )

    output = model(data)

    assert output.x.shape == x_prev.shape, "Node features shape mismatch"
    assert output.edge_attr.shape == edge_attr.shape, "Edge attributes shape mismatch"

    assert not torch.allclose(output.x, x_prev), "Node features were not modified"
    assert not torch.allclose(
        output.edge_attr, edge_attr_prev
    ), "Edge attributes were not modified"
