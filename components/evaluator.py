import numpy as np

from components.helpers import calculate_max_f1, calculate_f1

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, ndcg_score
from typing import Tuple
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def parse_data(data, device: str):
    return [torch.LongTensor(data[:, i].T).to(device=device) for i in range(data.shape[1])]


def _other_gene(gene_pair: Tuple[int, int], gene: int):
    return gene_pair[0] if gene_pair[1] == gene else gene_pair[1]


def _pad_zero(data, length=100):
    result = data + [0.0 for _ in range(length - len(data))]
    result = result[:length]
    return result


class Evaluator():
    NDCG_TOP_K_LIST = (10, 20, 50,)

    def __init__(self, params, model, data):
        self.params = params
        self.model = model
        self.data = data
        self.bce_loss_func = nn.BCELoss(reduction='none')

    def _init_data_loader(self, shuffle=True):
        return DataLoader(self.data, batch_size=self.params.batch_size, shuffle=shuffle)

    def _calc_roc_metrics(self, target_list, y_pred_list):
        _auc = roc_auc_score(target_list, y_pred_list)
        p, r, t = precision_recall_curve(target_list, y_pred_list)
        _aupr = auc(r, p)
        return _auc, _aupr

    def _calc_f1_metric(self, target_list, y_pred_list, force_f1_threshold: float = None):
        max_f1, max_f1_threshold = calculate_max_f1(target_list, y_pred_list)
        result = {
            'f1_score': max_f1, "f1_thres": max_f1_threshold,
            'max_f1_score': max_f1, "max_f1_thres": max_f1_threshold,
        }
        if force_f1_threshold is not None:
            result.update({
                "f1_thres": force_f1_threshold,
                'f1_score': calculate_f1(target_list, y_pred_list, force_f1_threshold),
            })
        return result

    def eval(self, return_metrics=True, force_f1_threshold: float = None):
        all_loss, batch_count = 0, 0
        all_target_list, all_y_pred_list = [], []
        all_sample_list, all_sample_results = [], []

        self.model.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(self._init_data_loader()):
                batch_count += 1
                head_ids, tail_ids, target_labels = parse_data(batch, self.params.device)
                score_pos, _other_loss = self.model(head_ids, tail_ids)

                target_labels_float = target_labels.unsqueeze(1).float().to(score_pos.device).expand(*score_pos.shape)
                loss_eval = self.bce_loss_func(score_pos, target_labels_float)
                loss_eval = torch.mean(loss_eval)

                target = target_labels.to('cpu').numpy().flatten().tolist()
                if len(target) == sum(target) or sum(target) <= 0:
                    continue

                all_loss += loss_eval.cpu().detach().numpy().item()
                y_pred = torch.mean(score_pos, dim=1).cpu().flatten().tolist()

                all_target_list.extend(target)
                all_y_pred_list.extend(y_pred)

                head_ids_cpu = head_ids.cpu().numpy().flatten().tolist()
                tail_ids_cpu = tail_ids.cpu().numpy().flatten().tolist()

                if not return_metrics:
                    self.model._is_frozen = False
                    score_pos, _ = self.model(head_ids, tail_ids)
                    score_pos_list = [score_pos[:, i].cpu().flatten().tolist() for i in range(score_pos.shape[1])]
                    self.model._is_frozen = True
                    for _row in zip(head_ids_cpu, tail_ids_cpu, target, y_pred, *score_pos_list):
                        all_sample_results.append(list(_row))

                all_sample_list.extend([tuple(x) for x in zip(head_ids_cpu, tail_ids_cpu)])

        metric_dict = {'loss': all_loss / batch_count}
        metric_dict["auc"], metric_dict["aupr"] = self._calc_roc_metrics(all_target_list, all_y_pred_list)
        metric_dict.update(self._calc_f1_metric(all_target_list, all_y_pred_list, force_f1_threshold))

        if return_metrics:
            result = metric_dict
        else:
            result = {"sample_list": all_sample_results, "metrics": metric_dict}
            print(metric_dict)
        return result
