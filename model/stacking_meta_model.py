from typing import Callable
import torch.nn as nn
import torch


class MyStackingMetaModel(nn.Module):
    def __init__(self, params, feat_dict: dict):
        super().__init__()
        # params
        self.params = params
        self.feat_dict = feat_dict

        # funcs
        self.sigmoid_func = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.model_pred_layer_1 = nn.Linear(4, 16)
        self.model_pred_layer_2 = nn.Linear(16, 1)
        self.model_pred_bn = nn.BatchNorm1d(16)

    def _build_feat_mat(self, head_ids, tail_ids):
        head_ids_cpu = [x.cpu().numpy().item() for x in head_ids]
        tail_ids_cpu = [x.cpu().numpy().item() for x in tail_ids]

        return torch.stack([
            self.feat_dict[(head_id, tail_id)]
            for head_id, tail_id in zip(head_ids_cpu, tail_ids_cpu)
        ]).to(self.params.device)

    def forward(self, head_ids, tail_ids):
        feat_mat = self._build_feat_mat(head_ids, tail_ids)

        # input shape : (batch_size, sub_model_count)
        # out shape   : (batch_size, 1)
        result = feat_mat
        result = self.relu(self.model_pred_bn(self.model_pred_layer_1(result)))
        result = self.model_pred_layer_2(result)
        result = self.sigmoid_func(result)

        return result, None

    def map_forward(self, head_ids, tail_ids, map_func: Callable):
        feat_mat = self._build_feat_mat(head_ids, tail_ids)
        return map_func(feat_mat)
