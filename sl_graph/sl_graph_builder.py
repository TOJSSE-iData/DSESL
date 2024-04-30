from model.data_loader import load_data_file
from components.helpers import root_abs_path

from torch_geometric.data import HeteroData
import numpy as np
import torch


def _cosine_similarity(matrix):
    X = matrix
    alpha = torch.sum(X * X, dim=1)
    norm = alpha.unsqueeze(1) * alpha.unsqueeze(0)
    index = torch.where(norm == 0)
    norm[index] = 1
    similarity_matrix = torch.mm(X, X.T) / (torch.sqrt(norm))
    similarity_matrix[index] = 0
    result = similarity_matrix
    result = result - torch.diag(torch.diag(result))
    result = torch.abs(result)
    return result


class BuildSLGraph:
    ID_MAPPING_PATH = root_abs_path("data_new/gene-id-mapping.csv")

    def __init__(self, train_data: np.array, transe_embeddings, device, gene_feat_dict: dict = None):
        self.train_data = train_data
        self.transe_embeddings = transe_embeddings
        self.gene_feat_dict = gene_feat_dict
        self.device = device
        # values
        self.id2index_dict, self.gene_id_list = self.__init_id_index_dict()
        self.node_feat_dim = -1

    def __init_id_index_dict(self):
        id2index_dict, index2id_dict = dict(), []
        for gene_id, name, ncbi_id in load_data_file(self.ID_MAPPING_PATH, split_char=",", to_int=False):
            index = len(index2id_dict)
            gene_id = int(gene_id)
            id2index_dict[gene_id] = index
            index2id_dict.append(gene_id)
        return id2index_dict, index2id_dict

    def _build_gene_feat_mat(self):
        result = []
        for _index, _gene_id in enumerate(self.gene_id_list):
            feat_row = self.transe_embeddings[_gene_id]
            if self.gene_feat_dict:
                feat_row = torch.cat([feat_row, self.gene_feat_dict[_gene_id].to(self.device)])
            result.append(feat_row)
        result = torch.stack(result).to(self.device)
        self.node_feat_dim = result.shape[1]
        return result

    def _build_edge_index(self, target_label: int):
        assert target_label in (0, 1)
        result = []
        for gene_a_id, gene_b_id, label in self.train_data:
            if label != target_label:
                continue
            result.append([self.id2index_dict[gene_a_id], self.id2index_dict[gene_b_id]])
            result.append([self.id2index_dict[gene_b_id], self.id2index_dict[gene_a_id]])
        result = np.array(result, dtype=int).T
        result = torch.LongTensor(result).to(self.device)
        return result

    def _build_pyg_graph(self) -> HeteroData:
        # nodes
        result = HeteroData()
        result["gene"].x = self._build_gene_feat_mat()
        # undirected edges
        result["gene", "gg_sl", "gene"].edge_index = self._build_edge_index(target_label=1)
        result["gene", "gg_sr", "gene"].edge_index = self._build_edge_index(target_label=0)
        return result

    def run(self):
        sl_graph = self._build_pyg_graph()
        return self.id2index_dict, sl_graph

    def build_similarity_mat(self, graph: HeteroData):
        result = _cosine_similarity(graph["gene"].x)
        return result
