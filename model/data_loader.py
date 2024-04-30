from components.helpers import root_abs_path

from typing import Dict
import numpy as np
import torch

ENTITY_COUNT = 47764
WHITE_KGE_TYPE_LIST = (
    "TransE_l2", "TransR", "DistMult", "RESCAL", "ComplEx", "RotatE",
    "TransE_l2_256",
)


def load_entity_embeddings(device, kge_type: str) -> torch.Tensor:
    # shape: (node_count, 64)
    assert kge_type in WHITE_KGE_TYPE_LIST
    print(f"[debug] kge_method={kge_type}")
    kg_embed = np.load(root_abs_path(f'data_new/dgl-ke/sl_kg_{kge_type}_entity_remapped.npy'))
    result = torch.FloatTensor(kg_embed).to(device=device)
    return result


def load_gene_omics_feature_dict(device) -> Dict:
    # gene feat-mat: (6473, 862)
    _path = "data_new/gene-feat/protein_go-mat_remapped.npy"
    result = np.load(root_abs_path(_path), allow_pickle=True).item()
    result = {
        x: torch.FloatTensor(np.array(y, dtype=np.float16)).to(device)
        for x, y in result.items()
    }
    return result


def load_data_file(path: str, split_char=" ", to_int=True):
    map_func = int if to_int else (lambda x: x)
    result = []
    with open(path, "rt") as f:
        for line in f.readlines():
            if not line.strip():
                continue
            result.append([map_func(x.strip()) for x in line.strip().split(split_char) if x.strip()])
    result = np.array(result)
    return result
