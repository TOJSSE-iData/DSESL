from torch_geometric.nn import HeteroConv, GraphConv
import torch.nn as nn
import torch


class SLGraphGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, device, in_channels: int = -1):
        super().__init__()

        self.device = device
        self.relu = nn.ReLU()

        # Notice: GCNConv is only used for homogeneous graphs
        self.convs = torch.nn.ModuleList()
        for _channel_info in [(in_channels, hidden_channels), (hidden_channels, out_channels)]:
            conv_type_dict = dict()
            conv_type_dict[("gene", "gg_sl", "gene")] = GraphConv(*_channel_info, add_self_loops=False)
            conv_type_dict[("gene", "gg_sr", "gene")] = GraphConv(*_channel_info, add_self_loops=False)
            self.convs.append(HeteroConv(conv_type_dict, aggr='sum'))

        self.bn_list = torch.nn.ModuleList([
            nn.BatchNorm1d(x) for x in (hidden_channels, out_channels)
        ])

    def forward(self, graph):
        x_dict, edge_index_dict = graph.x_dict, graph.edge_index_dict
        for _conv, _bn in zip(self.convs, self.bn_list):
            x_dict = _conv(x_dict, edge_index_dict)
            x_dict = {key: self.relu(_bn(x)) for key, x in x_dict.items()}
        return x_dict
