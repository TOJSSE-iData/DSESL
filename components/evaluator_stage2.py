from components.evaluator import Evaluator, parse_data

from typing import Callable
import torch


class Stage2Evaluator(Evaluator):

    def eval_sub_models(self, pred_map_func: Callable, force_f1_threshold: float = None):
        all_target_list, all_y_pred_list = [], []

        self.model.eval()

        with torch.no_grad():
            for b_idx, batch in enumerate(self._init_data_loader(shuffle=True)):
                head_ids, tails_ids, target_labels = parse_data(batch, self.params.device)
                score_pos = self.model.map_forward(
                    head_ids, tails_ids, map_func=pred_map_func,
                )

                target = target_labels.to('cpu').numpy().flatten().tolist()
                if len(target) == sum(target) or sum(target) <= 0:
                    continue
                y_pred = torch.mean(score_pos, dim=1).cpu().flatten().tolist()

                all_target_list.extend(target)
                all_y_pred_list.extend(y_pred)

        metric_dict = {'loss': -1}
        metric_dict["auc"], metric_dict["aupr"] = self._calc_roc_metrics(all_target_list, all_y_pred_list)
        metric_dict.update(self._calc_f1_metric(all_target_list, all_y_pred_list, force_f1_threshold))

        return metric_dict
