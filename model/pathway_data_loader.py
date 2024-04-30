from components.helpers import root_abs_path, load_json_file

from torch_geometric.data import Data
from typing import Tuple
import numpy as np
import torch


def ncbi_gene_mapping_dict(id_mapping_path: str):
    ncbi_id_dict = dict()
    with open(id_mapping_path, "rt") as f:
        for line in f.readlines():
            line = line.split(",")
            # {ncbi_id => kg_id}
            ncbi_id_dict[int(line[2])] = int(line[0])
    return ncbi_id_dict


class LoadPathwayData:
    NCBI_INFO_PATH = root_abs_path("data_pathway/ncbi-info.json")
    LOCATION_PATH = root_abs_path("data_pathway/pathway-location-info.json")
    TEXT_EMBED_PATH = root_abs_path("data_pathway/pathway-embedding.json")
    GRAPH_MAT_PATH = root_abs_path("data_pathway/graph-mat.pt.npy")

    ID_MAPPING_PATH = root_abs_path("data_new/gene-id-mapping.csv")
    PATHWAY_TREE_PATH = root_abs_path("data_pathway/pathway-tree.json")

    def __init__(self):
        self.ncbi_dict = load_json_file(self.NCBI_INFO_PATH)
        self.pathway_list = list(sorted(self.ncbi_dict.keys()))

    def _load_location_info(self) -> np.array:
        # 10 dim
        result = np.zeros((len(self.pathway_list), 10), dtype=int)
        for key, value in load_json_file(self.LOCATION_PATH).items():
            _index = self.pathway_list.index(key)
            value = np.array(list(value) + [0 for _ in range(10 - len(value))], dtype=int)
            result[_index, :] = value
        # normalization
        result = (result - result.min(axis=0)) / (result.max(axis=0) - result.min(axis=0))
        return result

    def _load_text_embedding(self) -> np.array:
        # 384 dim
        result = np.zeros((len(self.pathway_list), 384), dtype=float)
        for key, value in load_json_file(self.TEXT_EMBED_PATH).items():
            _index = self.pathway_list.index(key)
            value = np.array(list(value), dtype=float)
            result[_index, :] = value
        return result

    def _load_graph_mat(self):
        data = np.load(self.GRAPH_MAT_PATH)
        nonzero_index = np.nonzero(data)
        edge_index = np.array(nonzero_index, dtype=int)
        edge_weight = []
        for i, j in zip(*nonzero_index):
            edge_weight.append(data[i][j])
        edge_weight = np.array(edge_weight, dtype=int)
        return edge_index, edge_weight

    def _build_tree_edge_info(self):
        result = []
        for child_name, parent_name in load_json_file(self.PATHWAY_TREE_PATH).items():
            if not (child_name in self.pathway_list and parent_name in self.pathway_list):
                continue
            _edge = [self.pathway_list.index(child_name), self.pathway_list.index(parent_name)]
            result.append(_edge)
            result.append(_edge[::-1])
        edge_index = np.array(result, dtype=int).T
        edge_index = torch.LongTensor(edge_index)
        return edge_index

    def build_pyg_graph(self, device) -> Tuple[Data, Data]:
        # (2137, 394)
        x_data = np.concatenate(
            (self._load_location_info(), self._load_text_embedding()), axis=1, dtype=float,
        )
        x_data = torch.FloatTensor(x_data).to(device)
        # gene link
        edge_index, edge_weight = self._load_graph_mat()
        return (
            Data(
                x=x_data, edge_index=torch.LongTensor(edge_index).to(device),
                edge_weight=torch.FloatTensor(edge_weight).to(device),
            ),
            Data(x=x_data, edge_index=torch.LongTensor(self._build_tree_edge_info()).to(device)),
        )

    def build_gene_dict(self):
        ncbi_id_dict = ncbi_gene_mapping_dict(self.ID_MAPPING_PATH)
        result = dict()
        for p_id, p_name in enumerate(self.pathway_list):
            for gene_ncbi_id in self.ncbi_dict[p_name]["gene_ids"]:
                if gene_ncbi_id not in ncbi_id_dict:
                    continue
                key = ncbi_id_dict[gene_ncbi_id]
                result[key] = result.get(key, [])
                result[key].append(p_id)
        return result
