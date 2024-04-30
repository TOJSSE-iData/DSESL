from model.data_loader import load_data_file, load_entity_embeddings, load_gene_omics_feature_dict
from model.ensemble_model import MyEnsembleModel
from struct_gnn.meta_path import MetaPathDataset
from model.pathway_data_loader import LoadPathwayData
from components.trainer import Trainer
from components.evaluator import Evaluator
from components.file_db import FileDB
from components.helpers import TLprint, root_abs_path

from torch.multiprocessing import Process, set_start_method, freeze_support
from time import time
from typing import Callable
import numpy as np
import argparse
import torch
import logging
import sys
import os

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="\n%(asctime)s;%(levelname)s;%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

try:
    set_start_method('spawn')
except RuntimeError as e:
    pass

print("[CUDA]", torch.cuda.is_available())
print("[CUDA]", torch.cuda.device_count())


def parse_cli_args(arg_wrap_func: Callable = None):
    parser = argparse.ArgumentParser(description='DSESL model CLI')
    parser.add_argument('--omics_dim', type=int, default=862, help='GO Omics feature dimension')
    parser.add_argument("--gene_dim", type=int, default=64, help="default Gene feature dimension")
    parser.add_argument("--kge_method", type=str, default="TransE_l2", help="KGE method")
    parser.add_argument("--dropout", type=float, default=0.2, help="Default dropout rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Running one device")
    parser.add_argument("--num_epochs", type=int, default=30, help="Max training epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--eval_interval", type=int, default=256, help="Evaluate model every `eval_interval` batches")
    parser.add_argument("--l2", type=float, default=1e-5, help="L2 Regularization")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--early_stop", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--CV_mode", type=int, default=1, help="CV Mode: 1,2,3")
    parser.add_argument("--inner_cv_list", type=str, default="1,", help="inner cv parts")
    parser.add_argument("--sub_model_names", type=str, required=True, help="sub-model names in DSESL")
    parser.add_argument("--other_loss_beta", type=float, default=0.02, help="similarity loss")

    if arg_wrap_func is not None:
        arg_wrap_func(parser)

    result = parser.parse_args()
    result.device = torch.device(result.device)
    result.sub_model_names = list(result.sub_model_names.split(","))
    result.sub_model_names.sort()

    return result


def build_model(params, train_data):
    pathway_loader = LoadPathwayData()
    return MyEnsembleModel(
        params, train_data=train_data,
        transe_embeddings=load_entity_embeddings(params.device, kge_type=params.kge_method),
        pathway_graph=pathway_loader.build_pyg_graph(params.device),
        gene_pathway_dict=pathway_loader.build_gene_dict(),
        meta_path_data_dict=MetaPathDataset.load_feat_dict(norm_type="NPC", device=params.device),
        omics_embeddings=load_gene_omics_feature_dict(params.device),
    ).to(device=params.device)


def one_cv_part(params, data_dir, cv_part, file_db: FileDB):
    train_data = load_data_file(data_dir + "train.txt")
    model = build_model(params, train_data)
    metrics = Trainer(
        params, model,
        task_label=f"ensemble_{','.join(params.sub_model_names)}",
        cv_label=f"C{params.CV_mode}_cv{cv_part}",
        train_data=train_data,
        valid_evaluator=Evaluator(params, model, data=load_data_file(data_dir + "dev.txt")),
        test_evaluator=Evaluator(params, model, data=load_data_file(data_dir + "test.txt")),
    ).run()
    print(f"cv_part={cv_part} metric:", metrics)
    file_db.reload_from_disk()
    file_db.set(cv_part, metrics, flush=True)


def parse_metric_result(params, file_db: FileDB, suffix: str = None):
    METRIC_KEYS = ("loss", "auc", "aupr", "f1_score", "max_f1_score")
    metric_list = [
        y for x, y in file_db.get_all(flush=True).items()
        if suffix is None or str(x).endswith(suffix)
    ]
    TLprint(f"metrics for CV_mode={params.CV_mode}, inner_cv_list={params.inner_cv_list}, suffix={suffix or ''}")
    for i, _type in enumerate(("valid", "test")):
        for key in METRIC_KEYS:
            num_list = [float(x[i][key]) for x in metric_list]
            print(f"{_type}\t{key: <13}" + "\t{:.4f}±{:.4f}".format(
                np.mean(num_list), np.std(num_list),
            ))
    print(metric_list)


if __name__ == '__main__':
    freeze_support()

    params = parse_cli_args()
    file_db = FileDB(root_abs_path(f"data_new/file_db/CV{params.CV_mode}_{time()}.json"))

    for cv_part in [int(x) for x in params.inner_cv_list.split(",") if x.strip()]:
        TLprint(f"CV_mode={params.CV_mode}, cv_part={cv_part}")
        data_dir = root_abs_path(f"data_new/C{params.CV_mode}/cv_{cv_part}/")
        print("parent Process ID", os.getpid())
        cp = Process(target=one_cv_part, args=(params, data_dir, cv_part, file_db))
        cp.start()
        print(f"child Process, name={cp.name}, id={cp.pid}")
        cp.join()
    parse_metric_result(params, file_db)
