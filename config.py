import numpy as np
# some important parameters

# INPUT_HB_TYPE = ['HbO-All-Lowmid-High'] #

# INPUT_HB_TYPE = ['HbO-right-VPC-classification',
#                 'HbO-right-STG-classification',
#                 'HbO-MPC-classification']
                    # 'HbO-left-DPC-classification',
                    # 'HbO-right-DPC-classification',
                    # 'HbO-left-PSFC-classification',
#         'HbO-right-PSFC-classification'
#         ]

# INPUT_HB_TYPE = ['pre_post_treatment_hamd_reduction_50'] # for prognosis
# INPUT_HB_TYPE = ['pre_post_treatment_hamd_reduction_50',
                #  'DMFC/hbo/pre_post_treatment_hamd_reduction_50',
                #  'DMFC/hbr/pre_post_treatment_hamd_reduction_50',
                #  'DMFC/hbt/pre_post_treatment_hamd_reduction_50'] # for prognosis

CURRENT_LOO = 0
# INPUT_HB_TYPE = ['pre_treatment_hamd_reduction_50',
#                  'pre_post_treatment_hamd_reduction_50'
#                  ] # for prognosis
# fnirs-depression-deeplearning/allData/
INPUT_HB_TYPE=['prognosis/pre_treatment_hamd_reduction_50']
SPECIFY_FOLD=9

# INPUT_HB_TYPE = ['prognosis/pretreatment_benchmarks'] or ['prognosis/pre_treatment_hamd_reduction_50'] # for prognosis or 'diagnosis'

# for example:
# pre_post_treatment_hamd_reduction_50


# INPUT_HB_TYPE = [
#                 'HbO-left-STG-classification',
#                 'HbO-right-STG-classification',
#                 'HbO-left-VPC-classification',
#                 'HbO-right-VPC-classification',
#                 'HbO-MPC-classification',
#                 ]

IS_USING_WANDB = False

MAX_EPOCHS = 1000


# left_to_do SVM_ZHIFEI, RSFC_DUAN, NMF_ZHONG
MODELS_NEED_PREPROCESS_DATA = ['chao_cfnn',
                               'wang_alex', 
                               'zhu_xgboost', 
                               'yu_gnn',
                               'li_svm']

# PREPROCESSED_HB_FOLD_PATH = './allData/data_for_reproducing_model/HbO-All-Lowmid-High/'
PREPROCESSED_HB_FOLD_PATH = './allData/prognosis/pretreatment_benchmarks/'

# DEFAULT_HB_FOLD_PATH = './allData/Output_npy/twoDoctor/' # for MDD classification, original diagnosis
DEFAULT_HB_FOLD_PATH = './allData/' # for ./allData/prognosis/ diagnosis


MONITOR_METRIC = 'accuracy'  #  'accuracy' or 'sensitivity' or 'f1_score'

PARAMETER = {
    'graphsage_transformer': {
        'hb_path': 'data.npy',
        'adj_path': 'neighbour_matrix.npy',
        'depth': np.random.choice([1, 2, 3, 4]),
        'n_layers': np.random.choice([2, 4, 6]),
        'FFN_units': np.random.choice([64, 128, 256]),
        'lr_factor': np.random.choice([1, 5, 10, 20, 50, 100, 200]),
        'l2_rate': np.random.choice([0.001, 0.005, 0.01]),
        'lstm_unit': np.random.choice([64, 128, 256])
    },
    'mvg_transformer': {
        'hb_path': 'data.npy',
        'adj_path': 'multiview_adj_matrix5.npy',
        'batch_size': 128,
        'mvg_layers_depth': 1,  # np.random.choice([1,2,3,4])
        'n_layers': 12,  # np.random.choice([2, 4, 6]),
        'FFN_units': 256,  # np.random.choice([64, 128, 256]),
        'lr_factor': 1,  # np.random.choice([1, 10, 50, 100]),
        'l2_rate': 0.001  # np.random.choice([0.001, 0.005, 0.01]),
    },
    'mgm_transformer': {
        'hb_path': 'data.npy',
        'adj_path': 'multiview_adj_matrix5.npy',
        'batch_size': 128,
        'n_layers': 12,  # np.random.choice([2, 4, 6]),
        'FFN_units': 256,  # np.random.choice([64, 128, 256]),
        'lr_factor': 1,  # np.random.choice([1, 10, 50, 100]),
        'l2_rate': 0.001,  # np.random.choice([0.001, 0.005, 0.01]),
        'lstm_unit': np.random.choice([64, 128, 256])
    },
    'cnn_transformer': {
        'hb_path': 'hb_data.npy',
    },
    'pre_post_cnn_transformer': {
        'hb_path': 'data.npy',
    },
    'gnn': {
        'hb_path': 'data.npy',
        'adj_path': 'adj_matrix.npy',
    },
    'comb_cnn': {
        'hb_path': 'data.npy',
    },
    'gnn_transformer': {
        'hb_path': 'hb_data.npy',
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.01,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6#np.random.choice([4, 8, 12]),
    },
    'graphformer': {
        'hb_path': 'data.npy',
        'adj_path': 'adj_matrix.npy',
        'd_model': np.random.choice([16, 32, 64]),
        'batch_size': np.random.choice([4, 8, 16, 64]),
        'n_layers': np.random.choice([4, 8, 12]),
    },
    'rggcnn_transformer': {
        'hb_path': 'data.npy',
        'adj_path': 'neighbour_matrix.npy',
        'rggcnn_layers': 4
    },    
    'gin_transformer': {
        'hb_path': 'data.npy',
        'adj_path': 'neighbour_matrix.npy',
    },
    'transformer': {
        'hb_path': 'data.npy',
    },
    'mgn_transformer': {
        'hb_path': 'data.npy',
        'adj_path': 'euclidean_mgn_matrix.npy',
        'batch_size': 128,
        'mvg_layers_depth': 1,  # np.random.choice([1,2,3,4])
        'n_layers': 12,  # np.random.choice([2, 4, 6]),
        'FFN_units': 256,  # np.random.choice([64, 128, 256]),
        'lr_factor': 1,  # np.random.choice([1, 10, 50, 100]),
        'l2_rate': 0.001  # np.random.choice([0.001, 0.005, 0.01]),
    },
    'dgi_transformer': {
        'hb_path': 'dgi_data.npy',
    },
    'zhu_xgboost': {
        'hb_path': 'data.npy',
    },
    'chao_cfnn': {
        'hb_path': 'data.npy',
        'activation': 'relu',
        'lr': 0.1,
    },
    'wang_alex': {
        'hb_path': 'data.npy',
        'lr': 0.001,
        'activation': 'relu'
    },
    'yu_gnn': {
        'hb_path': 'data.npy',
        'adj_path': 'adj_1.npy',
    },
    'li_svm':{
        'hb_path': 'data.npy',
    }
}
