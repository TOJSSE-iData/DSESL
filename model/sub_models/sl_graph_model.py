from sl_graph.sl_graph_builder import BuildSLGraph
from sl_graph.graph_model import SLGraphGNN
from model.sub_models.kge_model import _rebuild_feat_mat
from model.sub_models.pathway_model_multi import _cosine_similarity

from typing import List
import torch.nn as nn
import torch
import numpy as np


class SubSLGraphModel(nn.Module):
    SLG_HIDDEN_DIM = 256
    SLG_OUT_DIM = 64

    def __init__(self, params, train_data: np.array, transe_embeddings, gene_feat_dict: dict = None):
        super().__init__()

        # params
        self.params = params
        self.transe_embeddings = transe_embeddings
        self.gene_feat_dict = gene_feat_dict

        # funcs
        self.dropout = nn.Dropout(p=params.dropout)
        self.relu = nn.ReLU()
        self.mse_loss_func = nn.MSELoss()

        # SL graph module: GNN
        builder = BuildSLGraph(train_data, transe_embeddings, params.device, gene_feat_dict=gene_feat_dict)
        self.sl_graph_gene_dict, self.sl_graph_data = builder.run()
        self.sl_graph_gnn = SLGraphGNN(hidden_channels=self.SLG_HIDDEN_DIM, out_channels=self.SLG_OUT_DIM,
                                       device=params.device).to(params.device)

        # predictor
        self.fc_layer = nn.Linear(self.params.gene_dim * 2, 256)
        self.fc_layer_1 = nn.Linear(256, 128)
        self.fc_layer_2 = nn.Linear(128, 1)
        self.fc_bn_1 = nn.BatchNorm1d(256)
        self.fc_bn_2 = nn.BatchNorm1d(128)

        # values
        self._batch_slg_feat_mat = None
        self._batch_final_feat_mat = None

    def _get_sl_gene_graph_feat(self, graph_out_mat, gene_ids: List[int]):
        result = []
        for gene_id in gene_ids:
            result.append(graph_out_mat[self.sl_graph_gene_dict[gene_id]])
        result = torch.stack(result)
        return result

    def forward(self, head_ids, tail_ids, to_score: bool = True):
        head_ids_cpu = [x.cpu().numpy().item() for x in head_ids]
        tail_ids_cpu = [x.cpu().numpy().item() for x in tail_ids]

        slg_data = self.sl_graph_gnn(self.sl_graph_data)["gene"]

        # SL-graph feat
        head_graph_feat = self._get_sl_gene_graph_feat(slg_data, head_ids_cpu)
        tail_graph_feat = self._get_sl_gene_graph_feat(slg_data, tail_ids_cpu)

        fused_sl_graph_feat = torch.cat([head_graph_feat, tail_graph_feat], dim=1)

        # fusion
        fuse_feat = fused_sl_graph_feat

        # final predict
        final_feat = fuse_feat
        predict_out = self.fc_layer_2(self.relu(self.fc_bn_2(
            self.fc_layer_1(self.relu(self.fc_bn_1(self.fc_layer(self.dropout(fuse_feat)))))
        )))

        #
        self._batch_slg_feat_mat = _rebuild_feat_mat(
            head_ids_cpu, tail_ids_cpu,
            self._get_sl_gene_graph_feat(self.sl_graph_data["gene"].x, head_ids_cpu),
            self._get_sl_gene_graph_feat(self.sl_graph_data["gene"].x, tail_ids_cpu),
        )
        self._batch_final_feat_mat = _rebuild_feat_mat(
            head_ids_cpu, tail_ids_cpu, head_graph_feat, tail_graph_feat
        )

        if to_score:
            result = predict_out
        else:
            # result = torch.cat([final_feat, predict_out], dim=1)
            result = predict_out

        return result

    def calc_loss(self):
        return self.mse_loss_func(
            _cosine_similarity(self._batch_slg_feat_mat, self._batch_slg_feat_mat),
            _cosine_similarity(self._batch_final_feat_mat, self._batch_final_feat_mat),
        )
