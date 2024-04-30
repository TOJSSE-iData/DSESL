from components.helpers import create_dir_if_not_exists, root_abs_path, colorful_print
from components.evaluator import parse_data

from tqdm import tqdm
from sklearn import metrics
import numpy as np
import os
import logging

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.optim as optim
import torch


class Trainer():
    MODEL_SAVE_ROOT = root_abs_path("results")

    def __init__(
            self, params, model, train_data, test_evaluator, valid_evaluator,
            task_label: str, cv_label: str, double_dataset: bool = True,
    ):
        self.model = model
        self.params = params
        self.train_data = self.__init_train_data(train_data, double_dataset)
        self.test_evaluator = test_evaluator
        self.valid_evaluator = valid_evaluator
        self.task_label = task_label
        self.cv_label = cv_label

        self.optimizer = optim.Adam(
            list(self.model.parameters()), lr=params.lr, weight_decay=self.params.l2
        )
        self.bce_loss_func = nn.BCELoss()
        self.reset_training_state()

        self.updates_counter = 0
        self.best_val_result = {}
        self.best_test_result = {}

    def __init_train_data(self, data: np.array, double_dataset: bool):
        result = data
        if double_dataset:
            result = np.vstack((result, result[:, [1, 0, 2]]))
        return result

    def __calc_model_path(self) -> str:
        return os.path.join(
            self.MODEL_SAVE_ROOT,
            f"{self.task_label}/best_model_{self.cv_label}.pth",
        )

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self, frozen_status: bool = False):
        train_all_auc, train_all_aupr, train_all_f1 = [], [], []
        total_loss = 0

        dataloader = DataLoader(
            self.train_data, shuffle=True,
            batch_size=self.params.batch_size * (32 if frozen_status else 1),
        )
        self.model.train()
        p_bar = tqdm(enumerate(dataloader))

        batch_count = 0
        for b_idx, batch in p_bar:
            head_ids, tails_ids, target_labels = parse_data(batch, self.params.device)
            self.optimizer.zero_grad()

            score_pos, _other_loss = self.model(head_ids, tails_ids)
            target_labels_float = target_labels.unsqueeze(1).float().to(score_pos.device).expand(*score_pos.shape)
            loss_train = self.bce_loss_func(score_pos, target_labels_float)

            if _other_loss is None:
                loss = loss_train
            else:
                _beta = self.params.other_loss_beta
                loss = loss_train * (1 - _beta) + _other_loss * _beta

            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()

            self.updates_counter += 1
            batch_count = b_idx

            # calculate train metric
            p_bar.set_description(f'batch: {b_idx}, loss_train={loss.cpu().detach().numpy()}')
            with torch.no_grad():
                total_loss += loss.item()
                target = target_labels.to('cpu').numpy().flatten().tolist()
                y_preds = torch.mean(score_pos, dim=1).cpu().flatten().tolist()

                if len(target) == sum(target) or sum(target) <= 0:
                    continue
                pred_scores = [1 if i else 0 for i in (np.asarray(y_preds) >= 0.5)]
                train_auc = metrics.roc_auc_score(target, y_preds)
                p, r, t = metrics.precision_recall_curve(target, y_preds)
                train_aupr = metrics.auc(r, p)

                train_f1 = metrics.f1_score(target, pred_scores)

                train_all_auc.append(train_auc)
                train_all_aupr.append(train_aupr)
                train_all_f1.append(train_f1)

            # calculate valida and test metric
            if self.updates_counter % self.params.eval_interval == 0:
                self._make_valid_test()

        train_loss = total_loss / batch_count
        train_auc = np.mean(train_all_auc)
        train_aupr = np.mean(train_all_aupr)
        train_f1 = np.mean(train_all_f1)

        if frozen_status:
            self._make_valid_test()

        return train_loss, train_auc, train_aupr, train_f1

    def _run_once(self, frozen_status: bool = False):
        self.reset_training_state()

        _max_epoch = self.params.num_epochs
        if frozen_status:
            _max_epoch = max(_max_epoch, 3)

        for epoch in range(1, _max_epoch + 1):
            train_loss, train_auc, train_aupr, train_f1 = self.train_epoch(frozen_status)
            logging.info(
                f'frozen_status={frozen_status}, Epoch {epoch} with loss: {train_loss}, '
                f'training AUC: {train_auc}, training AUPR: {train_aupr}'
            )
            # early stop
            if self.not_improved_count > self.params.early_stop:
                break

        return self.best_val_result, self.best_test_result

    def _freeze_sub_models(self):
        with open(self.__calc_model_path(), "rb") as f:
            self.model = torch.load(f)
        self.model.freeze_sub_models()
        self.optimizer = optim.Adam(
            list(self.model.parameters()), lr=self.params.lr, weight_decay=self.params.l2
        )
        self.valid_evaluator.model = self.model
        self.test_evaluator.model = self.model

    def run(self):
        result = self._run_once()
        # if len(self.params.sub_model_names) >= 2:
        #     self._freeze_sub_models()
        #     result = self._run_once(frozen_status=True)
        return result

    def _make_valid_test(self):
        val_result = self.valid_evaluator.eval()
        test_result = self.test_evaluator.eval(force_f1_threshold=val_result["max_f1_thres"])

        logging.info(colorful_print(f'[Validation] {val_result}'))
        logging.info(colorful_print(f'[Test] {test_result}', color="yellow"))

        if val_result['auc'] >= self.best_metric:
            self._save_classifier()
            self.best_metric = val_result['auc']
            self.not_improved_count = 0
            self.best_val_result = val_result
            self.best_test_result = test_result
            logging.info(colorful_print(f'[Best test results] {test_result}', color="yellow"))
        else:
            self.not_improved_count += 1
            if self.not_improved_count > self.params.early_stop:
                logging.info(f"metrics not improve for {self.params.early_stop} epochs and training will stop.")
                logging.info(colorful_print(f'[Best test results] {self.best_test_result}', color="yellow"))

        self.last_metric = val_result['auc']

        return val_result, test_result

    def _save_classifier(self):
        dest_path = self.__calc_model_path()
        create_dir_if_not_exists(os.path.dirname(dest_path))
        torch.save(self.model, dest_path)
        logging.info(f'Save current best model weights to: {dest_path}')
        return dest_path
