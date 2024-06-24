from configs.config import *
import os 
INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 10
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
    
PARAMETER['gnn_transformer'] = {
    'hb_path': 'hbo_simple_data.npy',  # 'merge_feature.npy', # hb_data
    'l1_rate': 0.01,  # should be 0.01
    'l2_rate': 0.001,  # should be 0.001
    'd_model': 64,  # np.random.choice([16, 32, 64]),
    'batch_size': 64,  # np.random.choice([4, 8, 16, 64]),
    'n_layers': 6,  # np.random.choice([4, 8, 12]),
    'classweight1': 1,
}

for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'hbo_simple_data.npy'
    PARAMETER[model]['classweight1'] = 1
    