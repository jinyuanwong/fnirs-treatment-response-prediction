from configs.mdd_classification_jamba import *

INPUT_HB_TYPE = ['diagnosis_110_fabeha_dataset_hb_all']


SPECIFY_FOLD = 5
for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_simple_all_1d.npy'