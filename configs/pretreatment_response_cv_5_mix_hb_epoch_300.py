from configs.config import *


INPUT_HB_TYPE = ['prognosis_mix_hb/pretreatment_response']
SPECIFY_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 300



parameters_cv_5_mix_hb = {
        'hb_path': 'hb_data.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.001,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6,#np.random.choice([4, 8, 12]),
        'epochs': 350
    }

for key, value in parameters_cv_5_mix_hb.items():
    PARAMETER['gnn_transformer'][key] = value
    PARAMETER['cnn_gnn_transformer'][key] = value