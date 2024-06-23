from configs.config import *
import os 
INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 10
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
HOLD_OUT_DIV = 10

for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'hbo_simple_data.npy'
    
    