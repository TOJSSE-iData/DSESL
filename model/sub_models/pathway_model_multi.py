from components.helpers import root_abs_path, load_json_file
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from typing import List, Tuple

import torch.nn.functional as F
import torch.nn as nn
import torch


def _cosine_similarity(tensor1, tensor2):
    # (2137, 394) * (2137, 394) -> (2137, 2137)
    return F.cosine_similarity(tensor1.unsqueeze(1), tensor2.unsqueeze(0), dim=2)


def _cosine_similarity_2(x, y):
    norm_x = torch.norm(x, dim=1, keepdim=True)
    norm_y = torch.norm(y, dim=1, keepdim=True)

    dot_product = torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(-1)  # (2137, 1)

    return dot_product / (norm_x * norm_y)


class PathwayGCN(torch.nn.Module):
    def __init__(self, in_channels: int = 394, out_channels=64):
        super(PathwayGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 256)
        self.conv2 = GCNConv(256, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class SubPathwayModel(nn.Module):
    PATHWAY_ROOT_DICT_PATH = root_abs_path("data_new/pathway-g/pathway-root-dict.json")

    def __init__(self, params, pathway_graph: Tuple[Data, Data], gene_pathway_dict: dict):
        super().__init__()

        # params
        self.params = params

        self.pathway_graph_1, self.pathway_graph_2 = pathway_graph
        self.pathway_graph_1 = self.pathway_graph_1.to(device=self.params.device)
        self.pathway_graph_2 = self.pathway_graph_2.to(device=self.params.device)
        self.gene_pathway_dict = gene_pathway_dict

        # funcs
        self.dropout = nn.Dropout(p=params.dropout)
        self.relu = nn.ReLU()
        self.mse_loss_func = nn.MSELoss()
        self.pathway_gnn_1 = PathwayGCN().to(self.params.device)
        self.pathway_gnn_2 = PathwayGCN().to(self.params.device)

        # values
        _pathway_json_data = load_json_file(self.PATHWAY_ROOT_DICT_PATH)
        self.pathway_root_dict = {int(x): int(y) for x, y in _pathway_json_data["root_dict"].items()}
        self.pathway_root_count = _pathway_json_data["root_count"]
        self._g_out_mat_1, self._g_out_mat_2 = None, None

        # Pathway fusing MLP
        self.pathway_layer1 = nn.Linear(64 * self.pathway_root_count, 256)
        self.pathway_layer2 = nn.Linear(256, self.params.gene_dim)
        self.pathway_bn_1 = nn.BatchNorm1d(256)
        self.pathway_bn_2 = nn.BatchNorm1d(self.params.gene_dim)

        # predictor
        self.fc_layer = nn.Linear(self.params.gene_dim * 2, 256)
        self.fc_layer_1 = nn.Linear(256, 128)
        self.fc_layer_2 = nn.Linear(128, 1)

    def _get_gene_pathway_feat(self, graph_out_mat_1, graph_out_mat_2, gene_ids: List[int]):
        result = []
        for gene_id in gene_ids:
            pathway_root_dict = dict()
            for _pathway_i in self.gene_pathway_dict[gene_id]:
                key = self.pathway_root_dict[_pathway_i]
                pathway_root_dict[key] = pathway_root_dict.get(key, [])
                pathway_root_dict[key].append(graph_out_mat_1[_pathway_i])
                pathway_root_dict[key].append(graph_out_mat_2[_pathway_i])
            feat_row = []
            for _root in range(self.pathway_root_count):
                if _root not in pathway_root_dict:
                    feat_row.append(torch.zeros(64).to(self.params.device))
                else:
                    _pathway_feat_mat = torch.stack(pathway_root_dict[_root])
                    feat_row.append(torch.sum(_pathway_feat_mat, dim=0).to(self.params.device))
            feat_row = torch.cat(feat_row, dim=0).to(self.params.device)
            result.append(feat_row)
        result = torch.stack(result)
        return result

    def forward(self, head_ids, tail_ids, to_score: bool = True):
        head_ids_cpu = [x.cpu().numpy().item() for x in head_ids]
        tail_ids_cpu = [x.cpu().numpy().item() for x in tail_ids]

        pathway_g_data_1 = self.pathway_gnn_1(self.pathway_graph_1)
        pathway_g_data_2 = self.pathway_gnn_2(self.pathway_graph_2)
        self._g_out_mat_1 = pathway_g_data_1
        self._g_out_mat_2 = pathway_g_data_2

        # Pathway feat
        head_pathway_feat = self._get_gene_pathway_feat(pathway_g_data_1, pathway_g_data_2, head_ids_cpu)
        tail_pathway_feat = self._get_gene_pathway_feat(pathway_g_data_1, pathway_g_data_2, tail_ids_cpu)

        fuse_head_pathway_feat = self.relu(self.pathway_bn_1(self.pathway_layer1(head_pathway_feat)))
        fuse_head_pathway_feat = self.pathway_bn_2(self.pathway_layer2(fuse_head_pathway_feat))
        fuse_tail_pathway_feat = self.relu(self.pathway_bn_1(self.pathway_layer1(tail_pathway_feat)))
        fuse_tail_pathway_feat = self.pathway_bn_2(self.pathway_layer2(fuse_tail_pathway_feat))

        # fusion
        fuse_feat = torch.cat([
            fuse_head_pathway_feat, fuse_tail_pathway_feat,
        ], dim=1)

        # final predict
        final_feat = self.fc_layer_1(self.relu(self.fc_layer(self.dropout(fuse_feat))))
        predict_out = self.fc_layer_2(self.relu(final_feat))

        if to_score:
            result = predict_out
        else:
            # result = torch.cat([final_feat, predict_out], dim=1)
            result = predict_out

        return result

    def calc_loss(self):
        return self.mse_loss_func(
            _cosine_similarity_2(self.pathway_graph_1.x, self.pathway_graph_2.x),
            _cosine_similarity_2(self._g_out_mat_1, self._g_out_mat_2),
        )
