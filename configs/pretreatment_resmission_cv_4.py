from configs.config import *


INPUT_HB_TYPE = ['prognosis/pretreatment_remission']
SPECIFY_FOLD = 4
STRATIFIED_CV_TOTAL_TRAININING_TIME = 1


PARAMETER['gnn_transformer'] = {
        'hb_path': 'hb_data.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.001,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6#np.random.choice([4, 8, 12]),
    }