from configs.mdd_classification_jamba import *

INPUT_HB_TYPE = ['diagnosis514']


SPECIFY_FOLD = 2
for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_simple_all_1d.npy'