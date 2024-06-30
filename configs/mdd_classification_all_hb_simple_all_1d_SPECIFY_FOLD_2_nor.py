from configs.config import *
from configs.models_args.transformer_args import Transformer_ModelArgs
import os 
INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 2
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
HOLD_OUT_DIV = 10

# PARAMETER['gnn_transformer'] = {
#         'hb_path': 'hbo_simple_data.npy',# 'merge_feature.npy', # hb_data
#         'l1_rate': 0.01,# should be 0.01
#         'l2_rate': 0.001,# should be 0.001
#         'd_model': 64,#np.random.choice([16, 32, 64]),
#         'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
#         'n_layers': 6, #np.random.choice([4, 8, 12]),
#         'classweight1': 1,
#     },
# for model, val in PARAMETER.items():
#     PARAMETER[model]['hb_path'] = 'hbo_simple_data.npy'
#     PARAMETER[model]['classweight1'] = 1

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
    