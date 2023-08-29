import os
import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from logger import Logger


class RNrToMetadata(object):
    def __init__(self, excel_path):
        self._logger = Logger()
        self._df = pd.read_excel(excel_path)

    def map_id_to_pam50(self):
        tnbc = self.get_tnbc_unique_df()

        # Create an instance of LabelEncoder to convert PAM50 subtypes
        encoder = LabelEncoder()
        # Fit and transform the 'PAM50 subtypes' column
        tnbc.loc[:, 'Encoded_PAM50 subtype'] = encoder.fit_transform(tnbc['PAM50 subtype'])
        self._log_encoder_mapping(encoder)

        return tnbc.set_index('SCANB_PD_ID')['Encoded_PAM50 subtype'].to_dict()

    @staticmethod
    def _log_encoder_mapping(label_encoder):
        for label, category in enumerate(label_encoder.classes_):
            print("PAM50 " + str(category) + " = " + str(label))

    def get_slides_to_parse(self):
        valid_rnrs = self.map_id_to_pam50()
        return list(valid_rnrs.keys())

    def get_tnbc_unique_df(self):
        tnbc = self._df[self._df['TNBC'] == 1]  # Focusing only on positive TNBC cases
        duplicated_ids = tnbc.duplicated(subset=['SCANB_PD_ID'])  # Assuming SCANB_PD_ID is unique- otherwise, fail.
        if duplicated_ids.isin([True]).any():
            self._logger.info("Found duplicated SCANB_PD_ID values")
            raise RuntimeError("Duplicated columns found, abort")

        tnbc = tnbc.dropna(subset=['PAM50 subtype'])  # Drop rows with missing PAM50

        # PAM50 subtype slides aren't equally distributed, so we will stick with the 3 top subtypes
        tnbc = tnbc.drop(tnbc.loc[tnbc['PAM50 subtype'].isin(["Luminal A", "Luminal B", "Normal-like"])].index)
        return tnbc

    def create_pam50_random_train_test_ids(self, random_seed=42, test_size=0.1, tiles_directory=None):
        tnbc = self.get_tnbc_unique_df()

        # we want to filter out IDs with no matching slide
        if tiles_directory:
            potential_ids = {}
            for img in os.listdir(tiles_directory):
                potential_ids[img[:img.find(".")]] = 1

            print("Originally found: {} unique records".format(len(tnbc)))
            tnbc = tnbc[tnbc['SCANB_PD_ID'].isin(potential_ids.keys())]
            print("Final cohort includes: {} unique records".format(len(tnbc)))
        else:
            print("Found " + str(len(tnbc)) + " unique TNBC records")

        # different PAM50 subtypes are not equally distributed, so we will manually ensure
        # equal representation of each of PAM 50 subtype class.

        # setting the random seed parameter so we can reproduce the results
        random.seed(random_seed)

        train_ids = []
        test_ids = []

        pam50_subtypes = list(tnbc['PAM50 subtype'].unique())  # We expect 2 different subtypes

        for subtype in pam50_subtypes:
            subtype_ids = tnbc[tnbc['PAM50 subtype'] == subtype]['SCANB_PD_ID'].to_list()
            subtype_size = len(subtype_ids)
            random.shuffle(subtype_ids)

            test_count = int(subtype_size * test_size) + 1
            train_count = subtype_size - test_count

            test_ids += subtype_ids[train_count:]
            train_ids += subtype_ids[:train_count]

        print("Those are test_ids: ", test_ids)
        return train_ids, test_ids

    def split_train(self, train_ids, random_seed=42, val_size=0.1):
        random.seed(random_seed)
        random.shuffle(train_ids)

        val_count = int(len(train_ids) * val_size)
        train_count = len(train_ids) - val_count

        val_ids = train_ids[train_count:]
        train_ids = train_ids[:train_count]

        return train_ids, val_ids
