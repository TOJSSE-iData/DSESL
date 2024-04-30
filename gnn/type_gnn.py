from gnn.base_funcs import ENTITY_DICT, load_entity_type_dict, load_kg_neighbor_dict, _entity_set_from_triplet
from model.data_loader import load_entity_embeddings

from components.helpers import create_dir_if_not_exists, root_abs_path
from model.data_loader import load_data_file

from typing import List
import os.path
import logging
import torch
import pickle


class TypeGNNFeature:
    CACHE_ROOT = create_dir_if_not_exists(root_abs_path("data_new/type_hop1"))

    FULL_SAMPLE_FILENAME = "sample-all.txt"
    KG_DATA_NAME = "kg2id.txt"
    ENTITY_TYPE_NAME = "entity-type.txt"

    def __init__(self, data_root: str, node_embeddings, device):
        self.data_root = data_root
        self.node_embeddings = node_embeddings.to(device="cpu")
        self.device = device
        # data
        self.entity_type_dict = load_entity_type_dict(os.path.join(data_root, self.ENTITY_TYPE_NAME))
        self.neighbor_dict = load_kg_neighbor_dict(os.path.join(data_root, self.KG_DATA_NAME), self.entity_type_dict)
        # values
        self.result_dict = {x: dict() for x in ENTITY_DICT.values()}
        self.white_gene_ids = self._get_white_gene_ids()

        print("[debug][type-hop1] len(self.entity_type_dict)", len(self.entity_type_dict))
        print("[debug][type-hop1] len(self.neighbor_dict)", len(self.neighbor_dict))
        print("[debug][type-hop1] len(self.white_gene_ids)", len(self.white_gene_ids))

    def _feat_cache_path(self, entity_type: str) -> str:
        return os.path.join(self.CACHE_ROOT, f"type_feat_{entity_type}.p")

    def _get_white_gene_ids(self, limit=0):
        data_path = os.path.join(self.data_root, self.FULL_SAMPLE_FILENAME)
        result = set()
        for gene_a_id, gene_b_id, label in load_data_file(data_path, split_char="\t"):
            for _id in (gene_a_id, gene_b_id):
                result.add(_id)
            if limit > 0 and len(result) >= limit:
                break
        return result

    def _one_gene(self, gene_id: int):
        triplet_set = set(self.neighbor_dict[gene_id])
        entity_set = _entity_set_from_triplet(triplet_set)
        for entity_type_id, entity_type_name in ENTITY_DICT.items():
            _id_list = [x for x in entity_set if self.entity_type_dict[x] == entity_type_id]
            if len(_id_list) <= 0:
                self.result_dict[entity_type_name][gene_id] = torch.zeros((64 * 3))
            else:
                feat_mat = torch.stack([self.node_embeddings[x] for x in _id_list])
                self.result_dict[entity_type_name][gene_id] = torch.cat([
                    torch.mean(feat_mat, dim=0),
                    torch.min(feat_mat, dim=0)[0],
                    torch.max(feat_mat, dim=0)[0],
                ])
        logging.debug(f"type_feat, gene_id = {gene_id}")

    def build(self):
        for gene_id in self.white_gene_ids:
            self._one_gene(gene_id)
        for _type, _data in self.result_dict.items():
            with open(self._feat_cache_path(_type), "wb") as f:
                pickle.dump(_data, f)
            print(self._feat_cache_path(_type))

    def load_feat_dict(self, key_list: List[str]):
        result = dict()
        for key in key_list:
            with open(self._feat_cache_path(key), "rb") as f:
                result[key] = pickle.load(f)
                result[key] = {x: y.to(self.device) for x, y in result[key].items()}
        return result


if __name__ == '__main__':
    TypeGNNFeature(
        node_embeddings=load_entity_embeddings("cpu", kge_type="TransE_l2"),
        data_root=root_abs_path("data_new"), device="cpu"
    ).build()
