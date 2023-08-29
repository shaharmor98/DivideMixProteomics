import os
import random

import numpy as np
import pandas as pd

from configuration import Configuration


class DiaToMetadata(object):
    def __init__(self, dia_excel_path, metadata_excel_path, tiles_directory):
        if os.path.exists(Configuration.DIA_PARTIAL_GENES_FILE_PATH):
            print("Found existing partial df")
            self._dia_df = pd.read_excel(Configuration.DIA_PARTIAL_GENES_FILE_PATH)
        else:
            self._dia_df = pd.read_excel(dia_excel_path, usecols=range(300))
            self._dia_df.to_excel(Configuration.DIA_PARTIAL_GENES_FILE_PATH)
        self._metadata_df = pd.read_excel(metadata_excel_path)
        self.tiles_directory = tiles_directory

    def map_scanb_pd_id_to_rnr(self, pd_ids):
        rnrs = {}

        for pd_id in pd_ids:
            res = self._metadata_df[self._metadata_df['SCANB_PD_ID'] == pd_id]
            if len(res) != 1:
                raise RuntimeError("Can't parse pd_ids to rnr, failed for: {} id".format(pd_id))

            rnrs[pd_id] = res['RNr'].values[0]

        return rnrs

    def get_genes_with_complete_records(self, rnrs):
        target_column_names = []
        for rnr in rnrs.values():
            target_column_names.append("ProteinQuant_" + rnr)

        condition = self._dia_df[target_column_names].notnull().all(axis=1)
        full_genes = self._dia_df.loc[condition, ["Gene_symbol"] + target_column_names]
        full_genes = full_genes.reset_index()
        return full_genes

    def get_gene_normalized_protein_quant(self, gene_row):
        cols = [col for col in gene_row.columns if col.startswith("ProteinQuant_")]

        # normalized_cols = ["MinMaxNorm" + col for col in self._dia_df.columns if col.startswith("ProteinQuant_")]
        # row = self._dia_df.loc[self._dia_df.index == gene_index, cols]

        def normalize(row):
            return (row - row.min()) / (row.max() - row.min())

        gene_row = gene_row[cols].apply(normalize, axis=1)
        return gene_row

    def get_upper_lower_percentage(self, row):
        high_percentile = np.percentile(row, 80)
        low_percentile = np.percentile(row, 20)

        high_cols = row[row >= high_percentile].dropna(axis=1).columns
        low_cols = row[row <= low_percentile].dropna(axis=1).columns

        return high_cols, low_cols, high_percentile, low_percentile

    def get_tnbc_unique_df(self):
        tnbc = self._metadata_df[self._metadata_df['TNBC'] == 1]  # Focusing only on positive TNBC cases
        duplicated_ids = tnbc.duplicated(subset=['SCANB_PD_ID'])  # Assuming SCANB_PD_ID is unique- otherwise, fail.
        if duplicated_ids.isin([True]).any():
            print("Found duplicated SCANB_PD_ID values")
            raise RuntimeError("Duplicated columns found, abort")

        return tnbc

    def get_existing_slides_ids(self):
        potential_ids = {}
        for img in os.listdir(self.tiles_directory):
            potential_ids[img[:img.find(".")]] = 1

        return potential_ids

    def get_normalized_gene_records(self, gene_name):
        tnbc = self.get_tnbc_unique_df()
        ids = self.get_existing_slides_ids()
        tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(ids.keys())]
        rnrs = self.map_scanb_pd_id_to_rnr(tnbc['SCANB_PD_ID'].to_list())
        genes = self.get_genes_with_complete_records(rnrs)
        gene_row = genes.loc[genes['Gene_symbol'] == gene_name]
        if len(gene_row) != 1:
            raise RuntimeError("WTF just happened with {}".format(gene_name))

        normalized_row = self.get_gene_normalized_protein_quant(gene_row)
        high_cols, low_cols, _, _ = self.get_upper_lower_percentage(normalized_row)

        target_rnrs = [(record[record.find("_") + 1:], record) for record in high_cols.to_list() if
                       record.startswith('ProteinQuant_')]
        target_rnrs += [(record[record.find("_") + 1:], record) for record in low_cols.to_list() if
                        record.startswith('ProteinQuant_')]

        slide_ids = {}
        for rnr, row_column in target_rnrs:
            slide_id = list(filter(lambda x: rnrs[x] == rnr, rnrs))[0]
            slide_ids[slide_id] = normalized_row[row_column].values[0]

        return slide_ids

    def get_continuous_normalized_records(self, gene_name):
        tnbc = self.get_tnbc_unique_df()
        ids = self.get_existing_slides_ids()
        tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(ids.keys())]
        rnrs = self.map_scanb_pd_id_to_rnr(tnbc['SCANB_PD_ID'].to_list())
        genes = self.get_genes_with_complete_records(rnrs)
        gene_row = genes.loc[genes['Gene_symbol'] == gene_name]
        if len(gene_row) != 1:
            raise RuntimeError("WTF just happened with {}".format(gene_name))

        normalized_row = self.get_gene_normalized_protein_quant(gene_row)
        target_rnrs = [(record[record.find("_") + 1:], record) for record in normalized_row.columns.to_list() if
                       record.startswith('ProteinQuant_')]

        slide_ids = {}
        for rnr, row_column in target_rnrs:
            slide_id = list(filter(lambda x: rnrs[x] == rnr, rnrs))[0]
            slide_ids[slide_id] = normalized_row[row_column].values[0]

        return slide_ids

    def get_gene_slides_with_labels(self, gene_name):
        tnbc = self.get_tnbc_unique_df()
        ids = self.get_existing_slides_ids()
        tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(ids.keys())]
        rnrs = self.map_scanb_pd_id_to_rnr(tnbc['SCANB_PD_ID'].to_list())
        genes = self.get_genes_with_complete_records(rnrs)
        gene_row = genes.loc[genes['Gene_symbol'] == gene_name]
        if len(gene_row) != 1:
            raise RuntimeError("WTF just happened with {}".format(gene_name))

        normalized_row = self.get_gene_normalized_protein_quant(gene_row)
        high_cols, low_cols, _, _ = self.get_upper_lower_percentage(normalized_row)

        target_rnrs = [(record[record.find("_") + 1:], 1) for record in high_cols.to_list() if
                       record.startswith('ProteinQuant_')]
        target_rnrs += [(record[record.find("_") + 1:], 0) for record in low_cols.to_list() if
                        record.startswith('ProteinQuant_')]

        slide_ids = {}
        for rnr, label in target_rnrs:
            slide_id = list(filter(lambda x: rnrs[x] == rnr, rnrs))[0]
            slide_ids[slide_id] = label

        return slide_ids

    def analyze_genes(self):
        print("Starting with: {} rows".format(len(self._metadata_df)))
        tnbc = self.get_tnbc_unique_df()
        print("Continuing with {} tnbc slides".format(len(tnbc)))
        ids = self.get_existing_slides_ids()
        print("Found {} existing slides".format(len(ids)))
        tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(ids.keys())]
        print("Continuing with {} existing slides".format(len(tnbc)))
        rnrs = self.map_scanb_pd_id_to_rnr(tnbc['SCANB_PD_ID'].to_list())
        genes = self.get_genes_with_complete_records(rnrs)
        print("Found {} complete genes, start analysing".format(len(genes)))

        results = {}
        summary_results = {}

        for row_number in range(len(genes)):
            gene_row = genes.loc[genes.index == row_number]
            gene_name = gene_row['Gene_symbol'].values[0]
            print("Gene: {}".format(gene_name))
            normalized_row = self.get_gene_normalized_protein_quant(gene_row)
            high_cols, low_cols, high_percentile, low_percentile = self.get_upper_lower_percentage(normalized_row)
            results[gene_name] = [high_cols, low_cols]
            summary_results[gene_name] = [high_percentile, low_percentile]

        """
        dist_summary = {}                                              
        for k,v in summary_results.items():                                                       
            high, low = v[0], v[1]                                                    
            distance = high-low                                                        
            dist_summary[k] = [distance, high, low]
        sorted_dict = dict(sorted(dist_summary.items(), key=lambda x: x[1][0], reverse=True))
        """
        """
        * start with '*'
        * Gene: STAT1, distance: 0.40545945263553945; high: 0.5408551981801274; low: 0.13539574554458794
        * Gene: HLA-C, distance: 0.38670822471874; high: 0.48883242492037154; low: 0.10212420020163153
        * Gene: NFKB2, distance: 0.38429598669478443; high: 0.4768618062047525; low: 0.09256581950996805
        """
        return results, summary_results

    def random_shuffle(self, gene_slides_with_labels, test_proportion=0.1):
        high_instances = [slide for slide, label in gene_slides_with_labels.items() if label == 1]
        low_instances = [slide for slide, label in gene_slides_with_labels.items() if label == 0]

        test_size = int(len(high_instances) * test_proportion)

        random.seed(Configuration.SEED)
        random.shuffle(high_instances)
        random.shuffle(low_instances)

        train_ids = {}
        test_ids = {}

        for instance in high_instances[:test_size]:
            test_ids[instance] = 1
        for instance in high_instances[test_size:]:
            train_ids[instance] = 1
        for instance in low_instances[:test_size]:
            test_ids[instance] = 0
        for instance in low_instances[test_size:]:
            train_ids[instance] = 0

        return train_ids, test_ids

    def random_disjoint_shuffle(self, gene_slides_with_labels, test_proportion=0.1):
        high_instances = [slide for slide, label in gene_slides_with_labels.items() if label == 1]
        low_instances = [slide for slide, label in gene_slides_with_labels.items() if label == 0]

        test_size = int(len(high_instances) * test_proportion)

        random.seed(Configuration.SEED)
        random.shuffle(high_instances)
        random.shuffle(low_instances)

        high_train_ids = {}
        low_train_ids = {}
        high_test_ids = {}
        low_test_ids = {}

        for instance in high_instances[:test_size]:
            high_test_ids[instance] = 1
        for instance in high_instances[test_size:]:
            high_train_ids[instance] = 1
        for instance in low_instances[:test_size]:
            low_test_ids[instance] = 0
        for instance in low_instances[test_size:]:
            low_train_ids[instance] = 0

        return high_train_ids, low_train_ids, high_test_ids, low_test_ids

    def get_gene_normalized_quant(self, gene_name):
        tnbc = self.get_tnbc_unique_df()
        ids = self.get_existing_slides_ids()
        tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(ids.keys())]
        rnrs = self.map_scanb_pd_id_to_rnr(tnbc['SCANB_PD_ID'].to_list())
        genes = self.get_genes_with_complete_records(rnrs)
        gene_row = genes.loc[genes['Gene_symbol'] == gene_name]
        if len(gene_row) != 1:
            raise RuntimeError("WTF just happened with {}".format(gene_name))

        normalized_row = self.get_gene_normalized_protein_quant(gene_row)
        return normalized_row

    def get_rnrs(self):
        tnbc = self.get_tnbc_unique_df()
        ids = self.get_existing_slides_ids()
        tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(ids.keys())]
        rnrs = self.map_scanb_pd_id_to_rnr(tnbc['SCANB_PD_ID'].to_list())
        return rnrs

    def split_by_expression_level(self, gene_name):
        row = self.get_gene_normalized_quant(gene_name)
        rnrs = self.get_rnrs()
        high_percentile = np.percentile(row, 70)
        low_percentile = np.percentile(row, 30)

        high_cols = row[row >= high_percentile].dropna(axis=1).columns
        low_cols = row[row <= low_percentile].dropna(axis=1).columns
        ood_cols = row[(row > low_percentile) & (row < high_percentile)].dropna(axis=1).columns

        high_rnrs = [(record[record.find("_") + 1:], 1) for record in high_cols.to_list() if
                     record.startswith('ProteinQuant_')]
        ood_rnrs = [record[record.find("_") + 1:] for record in ood_cols.to_list() if
                    record.startswith('ProteinQuant_')]
        low_rnrs = [(record[record.find("_") + 1:], 0) for record in low_cols.to_list() if
                    record.startswith('ProteinQuant_')]

        ood_ids = []
        for rnr in ood_rnrs:
            ood_ids.append(list(filter(lambda x: rnrs[x] == rnr, rnrs))[0])

        extreme_ids = {}
        for rnr, label in high_rnrs + low_rnrs:
            slide_id = list(filter(lambda x: rnrs[x] == rnr, rnrs))[0]
            extreme_ids[slide_id] = label

        return extreme_ids, ood_ids
