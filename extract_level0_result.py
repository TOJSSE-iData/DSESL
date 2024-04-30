from model.data_loader import load_data_file, load_gene_omics_feature_dict
from training import parse_cli_args
from components.trainer import Trainer
from components.feat_extractor import FeatExtractor
from components.helpers import TLprint, root_abs_path, create_dir_if_not_exists
from gnn.type_gnn import TypeGNNFeature
from struct_gnn.meta_path import MetaPathDataset
from sl_graph.sl_graph_builder import BuildSLGraph
from model.pathway_data_loader import LoadPathwayData

from torch.multiprocessing import Process, set_start_method, freeze_support
import torch
import numpy as np
import os


def _add_arg_list(parser):
    parser.add_argument("--slkb_mode", default=False, action='store_true', help="SLKB mode")
    parser.add_argument("--reload_data", default=False, action='store_true', help="reload data from disk")


def build_model(params, task_label: str, cv_label: str, cv_part: int):
    model_path = os.path.join(
        Trainer.MODEL_SAVE_ROOT,
        f"{task_label}/best_model_{cv_label}.pth",
    )
    with open(model_path, "rb") as f:
        result = torch.load(f, map_location=params.device)
    if params.reload_data:
        #
        result.model_list[0].omics_embeddings = load_gene_omics_feature_dict(params.device)
        result.model_list[0].graph_feat_dict = TypeGNNFeature(
            node_embeddings=result.model_list[0].transe_embeddings,
            data_root=root_abs_path("data_new"), device=params.device,
        ).load_feat_dict(key_list=list(result.model_list[0].entity_types))
        #
        result.model_list[1].meta_path_data_dict = MetaPathDataset.load_feat_dict(norm_type="NPC", device=params.device)
        #
        slg_model = result.model_list[2]
        builder = BuildSLGraph(
            train_data=load_data_file(root_abs_path(f"data_new/C{params.CV_mode}/cv_{cv_part}/train.txt")),
            transe_embeddings=slg_model.transe_embeddings, device=params.device,
            gene_feat_dict=result.model_list[0].omics_embeddings,
        )
        slg_model.sl_graph_gene_dict, slg_model.sl_graph_data = builder.run()
        #
        pathway_loader = LoadPathwayData()
        result.model_list[3].gene_pathway_dict = pathway_loader.build_gene_dict()
        #
        result.params = params
        for _model in result.model_list:
            _model.params = params
    return result


def one_cv_part_extract(params, data_dir, task_label: str, cv_part, model_label: str = None):
    cv_label = f"C{params.CV_mode}_cv{cv_part}"
    print(f"task_label={task_label}, cv_label={cv_label}, model_label={model_label}")
    #
    model = build_model(params, task_label=model_label or task_label, cv_label=cv_label, cv_part=cv_part)
    feat_dict = FeatExtractor(params, model, data=load_data_file(data_dir + "test.txt")).eval()
    #
    dest_path = root_abs_path(f"data_new/level-0-feat/{task_label}/{cv_label}-feat-dict.npy")
    create_dir_if_not_exists(os.path.dirname(dest_path))
    np.save(dest_path, feat_dict, allow_pickle=True)
    print(f"save feat of {len(feat_dict)} gene pairs to: {dest_path}")
    return dest_path


if __name__ == '__main__':
    freeze_support()

    params = parse_cli_args(arg_wrap_func=_add_arg_list)

    model_label = f"ensemble_{','.join(params.sub_model_names)}"
    for cv_part in [int(x) for x in params.inner_cv_list.split(",") if x.strip()]:
        TLprint(f"CV_mode={params.CV_mode}, cv_part={cv_part}")
        print("parent Process ID", os.getpid())

        if params.slkb_mode:
            task_label = "ensemble_slkb"
            data_dir = root_abs_path(f"data_new/C4/cv_3/")
        else:
            task_label = model_label
            data_dir = root_abs_path(f"data_new/C{params.CV_mode}/cv_{cv_part}/")

        cp = Process(target=one_cv_part_extract, args=(params, data_dir, task_label, cv_part, model_label))
        cp.start()
        print(f"child Process, name={cp.name}, id={cp.pid}")
        cp.join()
