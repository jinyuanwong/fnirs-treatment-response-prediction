from configs.mdd_classification_jamba import *

INPUT_HB_TYPE = ['diagnosis514']

SPECIFY_FOLD = 3
HOLD_OUT_DIV = 4
for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_simple_all_1d.npy'