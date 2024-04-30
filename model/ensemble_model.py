from model.sub_models.kge_model import SubKgeModel
from model.sub_models.meta_path_model import SubMetaPathModel
from model.sub_models.sl_graph_model import SubSLGraphModel
from model.sub_models.pathway_model_multi import SubPathwayModel

import torch.nn as nn
import torch
import numpy as np


class MyEnsembleModel(nn.Module):
    MODEL_NAMES = ["kge", "meta_p", "slg", "pathway"]

    def __init__(
            self, params, train_data: np.array, transe_embeddings,
            pathway_graph, gene_pathway_dict: dict,
            meta_path_data_dict: dict, omics_embeddings: dict,
    ):
        super().__init__()
        # params
        assert len(params.sub_model_names) >= 1
        assert 0 == len(set(params.sub_model_names) - set(self.MODEL_NAMES))
        self.params = params

        # funcs
        self.sigmoid_func = nn.Sigmoid()
        self.relu = nn.ReLU()

        # sub-models
        model_list = [
            SubKgeModel(params, transe_embeddings, omics_embeddings),
            SubMetaPathModel(params, meta_path_data_dict),
            SubSLGraphModel(params, train_data, transe_embeddings, gene_feat_dict=omics_embeddings),
            SubPathwayModel(params, pathway_graph, gene_pathway_dict),
        ]
        self.model_list = nn.ModuleList([
            x for x, _name in zip(model_list, self.MODEL_NAMES)
            if _name in params.sub_model_names
        ])

        # ensemble strategy
        self.all_parameters = [_param for _model in self.model_list for _param in _model.parameters()]
        self._is_frozen = False
        self.model_pred_layer_1 = nn.Linear(1 * len(params.sub_model_names), 1)

    def freeze_sub_models(self):
        assert self._is_frozen == False
        for _param in self.all_parameters:
            _param.requires_grad = False
        self._is_frozen = True

    def unfreeze_sub_models(self):
        assert self._is_frozen == True
        for _param in self.all_parameters:
            _param.requires_grad = True
        self._is_frozen = False

    def forward(self, head_ids, tail_ids):
        result = []
        if self._is_frozen:
            for _model in self.model_list:
                result.append(_model(head_ids, tail_ids, to_score=False))
        else:
            for _model in self.model_list:
                result.append(self.sigmoid_func(_model(head_ids, tail_ids)))
        result = torch.cat(result, dim=1)

        other_loss = None
        for _model in self.model_list:
            if self._is_frozen or not hasattr(_model, "calc_loss"):
                continue
            if other_loss is None:
                other_loss = _model.calc_loss()
            else:
                other_loss += _model.calc_loss()

        return result, other_loss
