import os


class Configuration(object):
    SEED = 42
    GENES = ["MKI67"]
    CHECKPOINTS_PATH = os.path.join(os.path.expanduser("~"), "paper/checkpoints", "{gene}")
    PATCH_SIZE = 512
    ZOOM_LEVEL = 20
    TILES_DIRECTORY = os.path.join(os.path.expanduser("~"), "proteomics-tiles-shahar-mor",
                                   "images/zoom_{zoom_level}_size_{patch_size}")
    RNR_METADATA_FILE_PATH = os.path.join(os.path.expanduser("~"), "proteomics_from_janne", "Labels_etc",
                                          "230109_RNr_to_metadata.xlsx")
    DIA_GENES_FILE_PATH = os.path.join(os.path.expanduser("~"), "proteomics_from_janne", "Labels_etc",
                                       "SCANB_TNBC_diaPASEF_SN16_E104.xlsx")
    DIA_PARTIAL_GENES_FILE_PATH = os.path.join(os.path.expanduser("~"), "proteomics_from_janne", "Labels_etc",
                                               "partial.xlsx")
    PREDICTIONS_SUMMARY_FILE = os.path.join(CHECKPOINTS_PATH, "predictions.txt")
    OOD_FILE_PATH = os.path.join(CHECKPOINTS_PATH, "ood.txt")
    N_ROUNDS = 5
    BATCH_SIZE = 32
