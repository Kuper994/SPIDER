import os
from typing import Dict, Tuple, List, Union
import mygene
import numpy as np
import pandas as pd
from goatools import obo_parser
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_ncbi_associations


mg = mygene.MyGeneInfo()

def get_data_matrix(network: pd.DataFrame, ground_truth: pd.DataFrame = None, gene_expression: pd.DataFrame = None,
                    methods: Dict[Tuple, Tuple] = None, locations: pd.DataFrame = None,
                    co_locations: pd.DataFrame = None,
                    co_abundance: pd.DataFrame = None,
                    prot_expression: pd.DataFrame = None, gen_labels: bool = True):
    print('building data matrix')
    rows = []
    y = []
    edges = []
    relevant_genes = set(gene_expression.index) & set(locations.index)
    if prot_expression is not None:
        relevant_genes = relevant_genes & set(prot_expression.index)
    if gen_labels:
        true_edges = {(min(t.g1, t.g2), max(t.g1, t.g2)) for t in ground_truth.itertuples()}
    base_edges = list({(min(t.g1, t.g2), max(t.g1, t.g2)) for t in network.itertuples()})
    gene_expression = {t[0]: t[1] for t in gene_expression.itertuples()}
    if prot_expression is not None:
        prot_expression = {t[0]: t[1] for t in prot_expression.itertuples()}
    locations = {t[0]: t[1:] for t in locations.itertuples()}
    if co_locations is not None:
        co_locations = {tuple(sorted(t[0])): t[1] for t in co_locations.itertuples()}
    if co_abundance is not None:
        if not isinstance(co_abundance, dict):
            co_abundance = {tuple(sorted(t[0])): t[1] for t in co_abundance.itertuples()}
    for g1, g2 in base_edges:
        g1_val = g2_val = None
        g1_locs = g2_locs = None
        if g1 in relevant_genes:
            g1_val = gene_expression[g1]
            g1_locs = locations[g1]
        if g2 in relevant_genes:
            g2_val = gene_expression[g2]
            g2_locs = locations[g2]
        if g1_val is None or g2_val is None:
            continue
        if prot_expression is not None:
            p1_val = prot_expression[g1]
            p2_val = prot_expression[g2]
            sorted_values = sorted([(g1_val, g1_locs, p1_val, g1), (g2_val, g2_locs, p2_val, g2)], key=lambda t: t[0])
            res = [sorted_values[0][0], sorted_values[1][0], sorted_values[0][2], sorted_values[1][2]]
        else:
            sorted_values = sorted([(g1_val, g1_locs, g1), (g2_val, g2_locs, g2)], key=lambda t: t[0])
            res = [sorted_values[0][0], sorted_values[1][0]]
        if co_abundance is not None:
            if isinstance(co_abundance, dict):
                # if (g1, g2) not in co_abundance:
                #   continue
                co_p = co_abundance.get((g1, g2), None) or co_abundance.get((g2, g1), None)
                if co_p is None:
                    continue
                res.append(co_p)
            else:
                if g1 not in co_abundance.columns or g2 not in co_abundance.columns:
                    continue
                if np.isnan(co_abundance.loc[g1, g2].mean()):
                    continue
                co_p = co_abundance.loc[g1, g2]
                res.append(co_p)
        if locations is not None:
            res.extend(sorted_values[0][1] + sorted_values[1][1])
        if co_locations is not None:
            if (g1, g2) not in co_locations:
                continue
            co_loc = co_locations[(g1, g2)]
            res.append(co_loc)
        m = methods.get((g1, g2), None) or methods.get((g2, g1), None)
        if m is None:
            continue
        res.extend(list(m))
        rows.append(res)
        if gen_labels:
            y.append(int(tuple(sorted((g1, g2))) in true_edges))
        if prot_expression is not None:
            edges.append((sorted_values[0][3], sorted_values[1][3]))
        else:
            edges.append((sorted_values[0][2], sorted_values[1][2]))
    df = pd.DataFrame(rows)
    return df, y, edges


def create_locations_from_go():
    # wget http://current.geneontology.org/ontology/subsets/go.obo
    os.makedirs('inputs', exist_ok=True)
    go_obo = 'inputs/go.obo'
    go = obo_parser.GODag(go_obo)
    fin_gene2go = download_ncbi_associations()
    objanno = Gene2GoReader(fin_gene2go, godag=go, taxids=[9606])
    ns2assc = objanno.get_ns2assc()

    terms = pd.read_csv('mouse_inputs/LOCATE.txt', header=0, sep='\t').columns[2:]
    goid_2_name = {id: go[id].name for id in go.keys() if go[id].name in terms}

    all_go_ids = list(goid_2_name.keys())
    rows = []
    for protein_id, go_ids in sorted(ns2assc['CC'].items()):
        terms_vector = np.array([int(i in go_ids) for i in all_go_ids])
        if terms_vector.sum():
            terms_vector = terms_vector.astype(float) / terms_vector.sum()
        rows.append([protein_id] + list(terms_vector))
    locations = pd.DataFrame(rows, columns=['Gene'] + all_go_ids).set_index('Gene')
    locations.to_csv('inputs/go_locations.csv')


def min_max_normalize(data):
    min_point = data.min()
    max_point = data.max()
    data = (data - min_point) / (max_point - min_point)
    return data


def get_cell_expression_data(cell_line: str, expression_path: str, normalize: bool = True,
                             fold_change: bool = False) -> pd.DataFrame:
    if not os.path.exists(expression_path):
        raise FileNotFoundError(f'File: {expression_path} does not exist.')
    dir_name = os.path.dirname(expression_path)
    sample_info_path = os.path.join(dir_name, 'sample_info.csv')
    if not os.path.exists(sample_info_path):
        raise FileNotFoundError(f'File: {sample_info_path} does not exist.')
    sample_info = pd.read_csv(sample_info_path, header=0)[['DepMap_ID', 'stripped_cell_line_name']]
    try:
        cell_line_id = sample_info[sample_info['stripped_cell_line_name'] == cell_line].iloc[0, 0]
    except IndexError:
        raise KeyError(f'Cell line: {cell_line} is not a valid cell line.')
    expression_data = pd.read_csv(expression_path, header=0, index_col=0)
    if fold_change:
        avg_exp = pd.DataFrame(expression_data.mean(axis=0).rename(index=lambda k: int(k.split('(')[-1].split(')')[0])))
        avg_exp['Gene'] = avg_exp.index
        avg_exp = avg_exp.groupby(['Gene']).agg('mean').reset_index().set_index('Gene').iloc[:, 0]
    expression_data = expression_data.loc[cell_line_id]
    expression_data = pd.DataFrame(expression_data.rename(index=lambda k: int(k.split('(')[-1].split(')')[0])))
    expression_data['Gene'] = expression_data.index
    expression_data = expression_data.groupby(['Gene']).agg('mean').reset_index().set_index('Gene')
    expression_data.rename(columns={cell_line_id: cell_line}, inplace=True)
    if fold_change:
        expression_data['fc'] = expression_data.div(avg_exp, axis=0)
    if normalize:
        return min_max_normalize(expression_data)
    else:
        return expression_data


def to_entrez_id(key: str = None, keys: List = None) -> Union[List, str]:
    if key is None and keys is None:
        raise ValueError('Provide either a string or list to translate.')
    if key:
        try:
            return mg.getgene(key)['entrezgene']
        except KeyError:
            print('Did not find: ' + key)
            return key
    if keys:
        return [v['entrezgene'] for v in mg.getgene(keys, fields='entrezgene')]


def to_symbol(key: str = None, keys: List = None) -> Union[List, str]:
    if key is None and keys is None:
        raise ValueError('Provide either a string or list to translate.')
    if key:
        try:
            return mg.getgene(key)['symbol']
        except KeyError:
            print('Did not find: ' + key)
            return key
    if keys:
        return [v['symbol'] for v in mg.getgene(keys, fields='symbol')]
