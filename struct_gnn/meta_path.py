from gnn.base_funcs import ENTITY_DICT, RELATION_DICT, \
    _yield_read_raw_triplet, load_kg_neighbor_dict, load_entity_type_dict
from model.data_loader import load_data_file
from components.helpers import create_dir_if_not_exists, root_abs_path, get_abs_paths_in_folder

from typing import Dict, List, Tuple, Set, Union
from random import shuffle, choice
import numpy as np
import itertools
import logging
import os.path
import pickle
import torch

ENTITY_DICT_R = {y: x for x, y in ENTITY_DICT.items()}
GENE_CLASS_ID = 2
GENE_CLASS_NAME = "Gene"


def _sample_reduce_to_size(data_list: List, target_size: int, force_sample=False):
    raw_index_list = list(range(len(data_list)))
    # put back sampling
    if force_sample:
        index_list = [choice(raw_index_list) for _ in range(target_size)]
    # sampling not required
    elif len(data_list) <= target_size:
        index_list = raw_index_list
    # no return sampling
    else:
        shuffle(raw_index_list)
        index_list = raw_index_list[:target_size]
    result = set([data_list[x] for x in index_list])
    return result


def _np_division(x, y):
    return np.divide(x, y, out=np.zeros_like(x, dtype=np.float64), where=y != 0)


def meta_paths_from(entity_list: List[int], relation_lists: List[Union[List[int], Set[int]]]) -> List[Tuple]:
    result = []
    for _relations in itertools.product(*relation_lists):
        _meta_path = [entity_list[0]]
        for i, _relation in enumerate(_relations):
            _meta_path.extend([_relation, entity_list[i + 1]])
        result.append(tuple(_meta_path))
    return result


class MetaPathDataset:
    CACHE_ROOT = create_dir_if_not_exists(root_abs_path("data_new/struct_gnn_relation-cache"))

    FULL_SAMPLE_FILENAME = "sample-all.txt"
    SPACE_SAMPLE_FILENAME = "sample-space.txt"

    KG_DATA_NAME = "kg2id.txt"
    ENTITY_TYPE_NAME = "entity-type.txt"

    FEAT_EXPORT_FUNC_DICT = {
        # "H_NPC": lambda ht, h_all, t_all: _np_division(ht, np.sum(h_all, axis=0)),
        # "T_NPC": lambda ht, h_all, t_all: _np_division(ht, np.sum(t_all, axis=0)),
        "NPC": lambda ht, h_all, t_all: _np_division(ht, (np.sum(h_all, axis=0) + np.sum(t_all, axis=0))),
    }
    MAX_DISTANCE = 3

    def __init__(self, data_root: str, device):
        self.data_root = data_root
        self.device = device
        # data
        self.entity_type_dict = load_entity_type_dict(os.path.join(data_root, self.ENTITY_TYPE_NAME))
        self.neighbor_dict = load_kg_neighbor_dict(os.path.join(data_root, self.KG_DATA_NAME), self.entity_type_dict)
        self.kg_triplet_list = list(
            _yield_read_raw_triplet(os.path.join(data_root, self.KG_DATA_NAME), self.entity_type_dict))
        # values
        self.all_mata_path_types = self.__init_mata_path_types()
        self.raw_feat_cache_dir = create_dir_if_not_exists(os.path.join(self.CACHE_ROOT, "gene-pairs"))

    def __init_mata_path_types(self):
        # initialization
        entity_type_list = list(ENTITY_DICT.keys())
        relation_type_dict = dict()
        for rel_id, rel_info in RELATION_DICT.items():
            entity_a_id, entity_b_id = ENTITY_DICT_R[rel_info[0]], ENTITY_DICT_R[rel_info[2]]
            for key in {(entity_a_id, entity_b_id), (entity_b_id, entity_a_id)}:
                relation_type_dict[key] = relation_type_dict.get(key, set())
                relation_type_dict[key].add(rel_id)
        # possible meta-path types
        candidate_types = [(GENE_CLASS_ID, GENE_CLASS_ID)]
        for i in range(1, self.MAX_DISTANCE):
            for _item in itertools.permutations(entity_type_list, i):
                candidate_types.append((GENE_CLASS_ID, *_item, GENE_CLASS_ID))
        # consider edge types
        result = []
        for _entity_list in candidate_types:
            _relation_list = []
            _is_valid = True
            for i, _entity_id in enumerate(_entity_list[:-1]):
                key = (_entity_id, _entity_list[i + 1])
                _relation_list.append(relation_type_dict.get(key, []))
                _is_valid = _is_valid and key in relation_type_dict
            # impossible meta-path types
            if not _is_valid:
                continue
            # combinations of edge type
            result.extend(meta_paths_from(list(_entity_list), _relation_list))
        logging.info(f"all meta-path type count = {len(result)}, type_list={result}")
        return result

    def _one_gene_nodes(self, gene_id: int) -> Dict[int, Set]:
        result = {gene_id: set()}
        for _triplet in set(self.neighbor_dict[gene_id]):
            key = _triplet[2] if _triplet[0] == gene_id else _triplet[0]
            result[key] = result.get(key, set())
            result[key].add(_triplet[1])
        return result

    def _load_raw_feat_dict(self, need_filter=True):
        # load from cache
        result = dict()
        for cache_path in get_abs_paths_in_folder(self.raw_feat_cache_dir):
            with open(cache_path, "rb") as f:
                result.update(pickle.load(f))
        if not need_filter:
            return result
        # fiter by sample dataset
        data_path = os.path.join(self.data_root, self.SPACE_SAMPLE_FILENAME)
        new_result = dict()
        for gene_a_id, gene_b_id, label in load_data_file(data_path, split_char="\t"):
            key = (gene_a_id, gene_b_id)
            new_result[key] = result[key]
        result = new_result
        return result

    '''
    public
    '''

    def one_gene_pair(self, gene_a_id: int, gene_b_id: int) -> np.array:
        # searching for connectivity paths
        gene_a_node_dict = self._one_gene_nodes(gene_a_id)
        gene_b_node_dict = self._one_gene_nodes(gene_b_id)
        bridge_triplet = []
        for triplet in self.kg_triplet_list:
            if triplet[0] in gene_a_node_dict and triplet[2] in gene_b_node_dict:
                bridge_triplet.append(triplet)
            elif triplet[0] in gene_b_node_dict and triplet[2] in gene_a_node_dict:
                bridge_triplet.append(tuple(triplet[::-1]))
        # build path
        path_list = []
        for triplet in bridge_triplet:
            # record intermediate entities and relations
            _entity_list, _relation_list = [gene_a_id], []
            if triplet[0] == gene_a_id:
                _relation_list.append({triplet[1]})
                _entity_list.append(triplet[2])
            else:
                _relation_list.extend([gene_a_node_dict[triplet[0]], {triplet[1]}])
                _entity_list.extend([triplet[0], triplet[2]])
            if _entity_list[-1] != gene_b_id:
                _relation_list.append(gene_b_node_dict[triplet[2]])
                _entity_list.append(gene_b_id)
            # build possible paths
            _entity_list = [self.entity_type_dict[x] for x in _entity_list]
            path_list.extend(meta_paths_from(_entity_list, _relation_list))
        path_list = set(path_list)
        # count
        meta_path_count = dict()
        for key in path_list:
            meta_path_count[key] = meta_path_count.get(key, 0) + 1
        # return
        result = np.array([meta_path_count.get(x, 0) for x in self.all_mata_path_types])
        return result

    def cache_all(self, min_id: int = None, max_id: int = None, cache_by_gene=True):
        data_path = os.path.join(self.data_root, self.SPACE_SAMPLE_FILENAME)
        cache_path_func = lambda x: os.path.join(self.raw_feat_cache_dir, f"gene_a-{x}.p")
        # calc feature
        result = dict()
        cached_gene_a_set = set()
        for gene_a_id, gene_b_id, label in load_data_file(data_path, split_char="\t"):
            if min_id is not None and gene_a_id < min_id:
                continue
            if max_id is not None and gene_a_id > max_id:
                continue
            # verify the cache
            _cache_path = cache_path_func(gene_a_id)
            if gene_a_id not in cached_gene_a_set and os.path.exists(_cache_path):
                if not cache_by_gene:
                    with open(_cache_path, "rb") as f:
                        result.update(pickle.load(f))
                _is_cached = True
                cached_gene_a_set.add(gene_a_id)
            # cache by gene: all gene pairs containing gene_a have been cached
            if cache_by_gene and gene_a_id in cached_gene_a_set:
                continue
            # cache by gene-pair: check if (gene_a, gene_b) already exists
            if not cache_by_gene and (gene_a_id, gene_b_id) in result:
                continue
            result[(gene_a_id, gene_b_id)] = self.one_gene_pair(gene_a_id, gene_b_id)
            logging.info(f"create mata-path feature, id_pair=({gene_a_id}, {gene_b_id})")
        logging.info(f"total meta-path count={len(result)}")
        # write cache
        result_group = dict()
        for [gene_a_id, gene_b_id], value in result.items():
            key = gene_a_id
            result_group[key] = result_group.get(key, dict())
            result_group[key][(gene_a_id, gene_b_id)] = value
        for key, data_dict in result_group.items():
            with open(cache_path_func(key), "wb") as f:
                pickle.dump(data_dict, f)
        return result

    def export_feature(self):
        # read cache
        raw_feat_dict = self._load_raw_feat_dict()
        # group by gene_id
        gene_feat_dict = dict()
        for key, value in raw_feat_dict.items():
            for _id in key:
                gene_feat_dict[_id] = gene_feat_dict.get(_id, [])
                gene_feat_dict[_id].append(value)
        # build normalized features
        result = {x: dict() for x in self.FEAT_EXPORT_FUNC_DICT}
        data_path = os.path.join(self.data_root, self.FULL_SAMPLE_FILENAME)
        for norm_type, feat_func in self.FEAT_EXPORT_FUNC_DICT.items():
            for gene_a_id, gene_b_id, label in load_data_file(data_path, split_char="\t"):
                key = (gene_a_id, gene_b_id)
                result[norm_type][key] = feat_func(
                    raw_feat_dict[key],
                    gene_feat_dict[gene_a_id], gene_feat_dict[gene_b_id],
                )
            logging.info(f"export_norm_feat, norm_type={norm_type}")
        # save to disk
        for norm_type, value in result.items():
            with open(os.path.join(self.CACHE_ROOT, f"{norm_type}.p"), "wb") as f:
                pickle.dump(value, f)
        return result

    @staticmethod
    def load_feat_dict(norm_type: str, device):
        assert norm_type in MetaPathDataset.FEAT_EXPORT_FUNC_DICT
        cache_path = os.path.join(MetaPathDataset.CACHE_ROOT, f"{norm_type}.p")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
            result = {x: torch.FloatTensor(y).to(device) for x, y in result.items()}
            result.update({(gene_b, gene_a): y for (gene_a, gene_b), y in result.items()})
        return result

    def verify(self):
        raw_feat_dict = self._load_raw_feat_dict(need_filter=False)
        print("len(raw_feat_dict)", len(raw_feat_dict))
        print("raw_feat_dict.keys()[:10]", list(raw_feat_dict.keys())[:10])
        data_path = os.path.join(self.data_root, self.SPACE_SAMPLE_FILENAME)
        result = []
        for gene_a_id, gene_b_id, label in load_data_file(data_path, split_char="\t"):
            key = (gene_a_id, gene_b_id)
            if key in raw_feat_dict:
                continue
            logging.info(f"fail to verify gene_pair {key}")
            result.append(key)
        logging.info(f"verify fail count = {len(result)}, {len(set([x[0] for x in result]))}")
        return result


if __name__ == '__main__':
    import sys

    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # MetaPathDataset(data_root=root_abs_path("data_new"), device="cpu").verify()
    MetaPathDataset(data_root=root_abs_path("data_new"), device="cpu").export_feature()
