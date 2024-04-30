from components.helpers import root_abs_path
from model.sub_models.pathway_model_multi import _cosine_similarity
from gnn.base_funcs import ENTITY_DICT
from gnn.type_gnn import TypeGNNFeature

from typing import List
import torch.nn as nn
import torch


def _rebuild_feat_mat(head_ids, tail_ids, head_feat_mat, tail_feat_mat):
    result, id_set = [], set()
    for id_list, feat_mat in [
        (head_ids, head_feat_mat), (tail_ids, tail_feat_mat),
    ]:
        for i, _id in enumerate(id_list):
            if _id in id_set:
                continue
            id_set.add(_id)
            result.append(feat_mat[i])
    return torch.stack(result)


class ChannelAttention1d(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return nn.functional.softmax(out, dim=1)


class SubKgeModel(nn.Module):
    DEFAULT_E_TYPES = [
        "Pathway", "MolecularFunction", "Compound", "CellularComponent",
        "Disease", "PharmacologicClass",
    ]

    def __init__(
            self, params, transe_embeddings, omics_embeddings,
            entity_types: List[str] = None,
    ):
        super().__init__()

        entity_types = entity_types or self.DEFAULT_E_TYPES
        assert len(entity_types) > 0
        for _type in entity_types:
            assert _type in ENTITY_DICT.values()

        # params
        self.params = params
        self.transe_embeddings = transe_embeddings
        self.omics_embeddings = omics_embeddings
        self.entity_types = entity_types

        # funcs
        self.dropout = nn.Dropout(p=params.dropout)
        self.relu = nn.ReLU()
        self.mse_loss_func = nn.MSELoss()

        # Omics MLP
        self.mp_layer1 = nn.Linear(self.params.omics_dim + self.params.gene_dim, 256)
        self.mp_layer2 = nn.Linear(256, self.params.gene_dim)
        self.bn1 = nn.BatchNorm1d(256)

        # KGE embeddings
        self.kge_layer1 = nn.Linear(64, 128)
        self.kge_layer2 = nn.Linear(128, self.params.gene_dim)
        self.bn_kge = nn.BatchNorm1d(128)

        # Entity
        self.graph_feat_dict = TypeGNNFeature(
            node_embeddings=transe_embeddings,
            data_root=root_abs_path("data_new"), device=self.params.device,
        ).load_feat_dict(key_list=list(self.entity_types))
        self.entity_att = ChannelAttention1d(in_planes=len(self.entity_types))

        self.entity_layer1 = nn.Linear(len(self.entity_types) * 64 * 3, 256)
        self.entity_layer2 = nn.Linear(256, self.params.gene_dim)
        self.entity_bn = nn.BatchNorm1d(256)

        # predictor
        self.fc_layer = nn.Linear(self.params.gene_dim * 4, 256)
        self.fc_layer_1 = nn.Linear(256, 128)
        self.fc_layer_2 = nn.Linear(128, 1)

        # values
        self._batch_omics_feat_mat = None
        self._batch_final_feat_mat = None

    def get_omics_features(self, id_list):
        return torch.stack([self.omics_embeddings[x] for x in id_list]).to(device=self.params.device)

    def _get_entity_features(self, id_list):
        result = []
        for _type in self.entity_types:
            # (batch_size, 64 * 3)
            result.append(torch.stack([self.graph_feat_dict[_type][x] for x in id_list]))
        result = torch.stack(result, dim=1)
        entity_att = self.entity_att(result)
        result = result.mul(entity_att)
        result = result.view(result.shape[0], -1)
        #
        result = self.entity_layer2(self.relu(self.entity_bn(self.entity_layer1(result))))
        return result

    def forward(self, head_ids, tail_ids, to_score: bool = True):
        head_ids_cpu = [x.cpu().numpy().item() for x in head_ids]
        tail_ids_cpu = [x.cpu().numpy().item() for x in tail_ids]

        # KGE
        kge_embeddings = self.kge_layer2(self.bn_kge(self.relu(self.kge_layer1(self.transe_embeddings))))

        # OMICs feat
        head_omics_feat = self.get_omics_features(head_ids_cpu)
        tail_omics_feat = self.get_omics_features(tail_ids_cpu)

        fuse_head_omics_feat = self.mp_layer2(self.relu(self.bn1(self.mp_layer1(
            torch.cat([head_omics_feat, kge_embeddings[head_ids]], dim=1)
        ))))
        fuse_tail_omics_feat = self.mp_layer2(self.relu(self.bn1(self.mp_layer1(
            torch.cat([tail_omics_feat, kge_embeddings[tail_ids]], dim=1)
        ))))

        # GNN: entity
        fused_head_entity_feat = self._get_entity_features(head_ids_cpu)
        fused_tail_entity_feat = self._get_entity_features(tail_ids_cpu)

        # fusion
        fuse_feat = torch.cat([
            fuse_head_omics_feat, fuse_tail_omics_feat,
            fused_head_entity_feat, fused_tail_entity_feat,
        ], dim=1)

        # final predict
        final_feat = fuse_feat
        predict_out = self.fc_layer_2(self.relu(
            self.fc_layer_1(self.relu(self.fc_layer(self.dropout(fuse_feat))))
        ))

        #
        self._batch_omics_feat_mat = _rebuild_feat_mat(
            head_ids_cpu, tail_ids_cpu, head_omics_feat, tail_omics_feat
        )
        self._batch_final_feat_mat = _rebuild_feat_mat(
            head_ids_cpu, tail_ids_cpu, fuse_head_omics_feat, fuse_tail_omics_feat
        )

        if to_score:
            result = predict_out
        else:
            # result = torch.cat([final_feat, predict_out], dim=1)
            result = predict_out

        return result

    def calc_loss(self):
        return self.mse_loss_func(
            _cosine_similarity(self._batch_omics_feat_mat, self._batch_omics_feat_mat),
            _cosine_similarity(self._batch_final_feat_mat, self._batch_final_feat_mat),
        )
