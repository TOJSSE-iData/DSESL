from model.data_loader import load_data_file
from model.stacking_meta_model import MyStackingMetaModel
from model.ensemble_model import MyEnsembleModel
from training import parse_cli_args, parse_metric_result

from components.trainer import Trainer
from components.evaluator_stage2 import Stage2Evaluator
from components.file_db import FileDB
from components.helpers import TLprint, root_abs_path

from torch.multiprocessing import Process, set_start_method, freeze_support
from random import shuffle
from time import time
import torch.nn as nn
import numpy as np
import torch
import os

INNER_CV_COUNT = 5


def _load_feat_dict(params, cv_mode: int) -> dict:
    assert cv_mode in (1, 2, 3)
    data_path_f = root_abs_path("data_new/level-0-feat/ensemble_{}/C{}_cv{}-feat-dict.npy")
    _sub_model_names = ','.join(sorted(MyEnsembleModel.MODEL_NAMES))
    result = dict()
    for _inner_cv in range(1, 1 + INNER_CV_COUNT):
        _path = data_path_f.format(_sub_model_names, cv_mode, _inner_cv)
        print(f"data_path: {_path}")
        assert os.path.exists(_path)
        for key, value in np.load(_path, allow_pickle=True).item().items():
            result[key] = torch.FloatTensor(value).to(params.device)
    return result


def _load_stage2_dataset(data_dir, cv_part: int):
    assert cv_part in range(1, 1 + INNER_CV_COUNT)
    left_set_list = []
    test_set = None
    for _part in range(1, 1 + INNER_CV_COUNT):
        _sample_set = load_data_file(os.path.join(data_dir, f"cv_{_part}/test.txt"))
        if _part == cv_part:
            test_set = _sample_set
        else:
            left_set_list.append(_sample_set)
    # deduplication
    _test_sample_set = {tuple(x) for x in test_set}
    _left_sample_set = {tuple(x) for _set in left_set_list for x in _set if tuple(x) not in _test_sample_set}
    _left_sample_list = list(_left_sample_set)
    shuffle(_left_sample_list)
    _offset = int(7 / 8 * len(_left_sample_list))
    # validation set & testing set
    valid_set = np.array(_left_sample_list[_offset:])
    train_set = np.array(_left_sample_list[:_offset])
    print(
        f"[debug] cv_part={cv_part}, len(train, valid, test)", len(train_set), len(valid_set), len(test_set)
    )
    return train_set, valid_set, test_set


def _calc_sub_model_metrics(params, cv_part: int, valid_evaluator: Stage2Evaluator, test_evaluator: Stage2Evaluator):
    sigmoid_func = nn.Sigmoid()
    pred_func_dict = {
        "mean": (lambda x: torch.mean(sigmoid_func(x), dim=1).unsqueeze(1)),
    }
    sub_model_names = [x for x in MyEnsembleModel.MODEL_NAMES if x in params.sub_model_names]
    for model_name in sub_model_names:
        y_col = MyEnsembleModel.MODEL_NAMES.index(model_name)
        pred_func_dict[model_name] = (lambda x, i=y_col: sigmoid_func(x)[:, i].unsqueeze(1))
    #
    result = dict()
    for key, _func in pred_func_dict.items():
        valid_metric = valid_evaluator.eval_sub_models(_func)
        test_metric = test_evaluator.eval_sub_models(_func, force_f1_threshold=valid_metric["max_f1_thres"])
        result[key] = [valid_metric, test_metric]
    result = {f"{cv_part}_{x}": y for x, y in result.items()}
    return result


def build_model(params):
    return MyStackingMetaModel(
        params=params,
        feat_dict=_load_feat_dict(params, cv_mode=params.CV_mode)
    ).to(device=params.device)


def one_cv_part(params, data_dir, cv_part, file_db: FileDB):
    train_data, validation_data, test_data = _load_stage2_dataset(data_dir, cv_part)
    # initialization
    model = build_model(params)
    test_evaluator = Stage2Evaluator(params, model, data=test_data)
    valid_evaluator = Stage2Evaluator(params, model, data=validation_data)
    # sub-models
    sub_model_metrics = _calc_sub_model_metrics(params, cv_part, valid_evaluator, test_evaluator)
    print("sub_model_metrics=", sub_model_metrics)
    # meta-leaner
    metrics = Trainer(
        params, model,
        task_label=f"ensemble_stacking_{','.join(params.sub_model_names)}",
        cv_label=f"C{params.CV_mode}_cv{cv_part}",
        train_data=train_data, double_dataset=False,
        valid_evaluator=valid_evaluator, test_evaluator=test_evaluator,
    ).run()
    # record metrics to disk
    print(f"cv_part={cv_part} metric:", metrics)
    file_db.reload_from_disk()
    file_db.set(f"{cv_part}_level1", metrics, flush=False)
    file_db.update_many(sub_model_metrics, flush=True)


if __name__ == '__main__':
    freeze_support()

    params = parse_cli_args()
    file_db = FileDB(root_abs_path(f"data_new/file_db/CV{params.CV_mode}_{time()}.json"))

    for cv_part in [int(x) for x in params.inner_cv_list.split(",") if x.strip()]:
        TLprint(f"CV_mode={params.CV_mode}, cv_part={cv_part}")
        data_dir = root_abs_path(f"data_new/C{params.CV_mode}/")
        print("parent Process ID", os.getpid())
        cp = Process(target=one_cv_part, args=(params, data_dir, cv_part, file_db))
        cp.start()
        print(f"child Process, name={cp.name}, id={cp.pid}")
        cp.join()

    for suffix in (["_level1", "_mean"] + [f"_{x}" for x in params.sub_model_names]):
        parse_metric_result(params, file_db, suffix=suffix)
