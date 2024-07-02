from configs.config import *
from configs.models_args.transformer_args import Transformer_ModelArgs
import os 

from configs.mdd_classification_jamba import *
INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 4
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
HOLD_OUT_DIV = 5

transformer_args = Transformer_ModelArgs(
    batch_size=64,
    d_model=64,
    n_heads=1,
    class_weights={0: 1, 1: 1}, # for pretreament classification {0: 1, 1: 5}
    )

        
PARAMETER['cnn_transformer'] = {
    'hb_path': 'nor_hb_simple_all_1d.npy',
    'args': transformer_args,
}

for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_simple_all_1d.npy'
    PARAMETER[model]['classweight1'] = 1
    