import torch.nn as nn
import torch


class SubMetaPathModel(nn.Module):

    def __init__(self, params, meta_path_data_dict: dict):
        super().__init__()

        # params
        self.params = params
        self.meta_path_data_dict = meta_path_data_dict

        # funcs
        self.dropout = nn.Dropout(p=params.dropout)
        self.relu = nn.ReLU()

        # Meta-path
        self.mp_layer1_meta_p = nn.Linear(283, 128)
        self.mp_layer2_meta_p = nn.Linear(128, self.params.gene_dim * 2)
        self.bn_meta_p = nn.BatchNorm1d(128)

        # predictor
        self.fc_layer = nn.Linear(self.params.gene_dim * 2, 256)
        self.fc_layer_1 = nn.Linear(256, 128)
        self.fc_layer_2 = nn.Linear(128, 1)

    def forward(self, head_ids, tail_ids, to_score: bool = True):
        head_ids_cpu = [x.cpu().numpy().item() for x in head_ids]
        tail_ids_cpu = [x.cpu().numpy().item() for x in tail_ids]

        # Meta-path feat
        meta_path_feat_mat = []
        for head_id, tail_id in zip(head_ids_cpu, tail_ids_cpu):
            meta_path_feat_mat.append(self.meta_path_data_dict[(head_id, tail_id)])
        meta_path_feat = torch.stack(meta_path_feat_mat)
        meta_path_feat = self.mp_layer2_meta_p(self.bn_meta_p(self.relu(self.mp_layer1_meta_p(meta_path_feat))))

        # fusion
        fuse_feat = meta_path_feat

        # final predict
        final_feat = self.fc_layer_1(self.relu(self.fc_layer(self.dropout(fuse_feat))))
        predict_out = self.fc_layer_2(self.relu(final_feat))

        if to_score:
            result = predict_out
        else:
            # result = torch.cat([final_feat, predict_out], dim=1)
            result = predict_out

        return result
