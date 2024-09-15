import copy

import pandas as pd
import torch
from torch.utils.data import Dataset

from consts import DEVICE

"""
data should be a dataframe of the form:
[gene expression - g1, gene expression - g2, protein expression - g1, protein expression - g2, 
co-abundance, locations g1 ... ,locations g2 ..., co locations, methods, edge string (g1,g2)]
"""


class SPIDERDataset(Dataset):
    def __init__(self, data: pd.DataFrame, y_data: pd.DataFrame = None, expression_size: int = 2,
                 prot_size: int = 3, locations_size: int = 40, methods_size: int = 12, p_expression: bool = True,
                 co_abundance: bool = True):
        data = copy.deepcopy(data)
        data['g1'] = [e.split(', ')[0].split('(')[-1] for e in data.iloc[:, -1]]
        data['g2'] = [e.split(', ')[-1].split(')')[0] for e in data.iloc[:, -2]]
        if co_abundance:
            inter_idx = [expression_size + prot_size - 1] + \
                        list(range(expression_size + prot_size + locations_size - 1,
                                   expression_size + prot_size + locations_size + methods_size))
        else:
            inter_idx = list(
                range(expression_size + locations_size - 1, expression_size + locations_size + methods_size))
        interaction = data.iloc[:, inter_idx].astype(float).values
        node_mapping = {g: i for i, g in enumerate(list(set(data['g1']) | set(data['g2'])))}
        edge_index = data.loc[:, ['g1', 'g2']].applymap(lambda g: node_mapping.get(g)).values

        self._interaction = torch.tensor(interaction, dtype=torch.float32, requires_grad=True).to(DEVICE)
        self._edge_index = torch.tensor(edge_index).to(DEVICE)
        if p_expression:
            graph_matrix_g1 = data.iloc[:, [0, expression_size] + list(
                range(expression_size + prot_size, int(expression_size + prot_size + locations_size / 2))) + [-2]]
            graph_matrix_g2 = data.iloc[:, [1, expression_size + 1] + list(
                range(int(expression_size + prot_size + locations_size / 2),
                      int(expression_size + prot_size + locations_size - 1))) + [-1]]
        else:
            graph_matrix_g1 = data.iloc[:,
                              [0] + list(range(expression_size, int(expression_size + locations_size / 2))) + [-2]]
            graph_matrix_g2 = data.iloc[:, [1] + list(
                range(int(expression_size + locations_size / 2), int(expression_size + locations_size - 1))) + [-1]]
        graph_matrix_g1 = graph_matrix_g1.rename(columns={'g1': 'g'})
        graph_matrix_g2 = graph_matrix_g2.rename(columns={'g2': 'g'})
        graph_matrix_g1.columns = range(0, graph_matrix_g1.columns.size)
        graph_matrix_g2.columns = range(0, graph_matrix_g2.columns.size)
        graph_matrix = pd.concat((graph_matrix_g1, graph_matrix_g2)).drop_duplicates().set_index(
            graph_matrix_g1.columns.size - 1).rename(index=node_mapping)
        self._graph_matrix = torch.tensor(graph_matrix.sort_index().astype(float).values, dtype=torch.float32).to(
            DEVICE)
        self._node_mapping = node_mapping

        if y_data is not None:
            self._y_data = torch.tensor(y_data, dtype=torch.float32, requires_grad=True).to(DEVICE)
        else:
            self._y_data = None

    def __len__(self):
        return len(self._interaction)

    def __getitem__(self, idx):
        if self._y_data is not None:
            return (self._interaction[idx], self._edge_index[idx]), self._y_data[idx]
        else:
            return (self._interaction[idx], self._edge_index[idx]), self._y_data

    @property
    def interaction(self):
        return self._interaction

    @property
    def edge_index(self):
        return self._edge_index

    @property
    def graph_matrix(self):
        return self._graph_matrix

    @property
    def y(self):
        return self._y_data

    @property
    def node_mapping(self):
        return self._node_mapping
