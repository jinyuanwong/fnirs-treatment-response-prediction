from configs.config import *
from configs.models_args.mamba_args import ModelArgs
import os 
from configs.mdd_classification_mamba import *



INPUT_HB_TYPE = ['diagnosis_110_fabeha_dataset_hb_all']
SPECIFY_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
HOLD_OUT_DIV = 4

for model, model_args in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_simple_all_1d.npy'

