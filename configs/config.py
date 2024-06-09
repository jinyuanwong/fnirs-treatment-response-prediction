import numpy as np

INPUT_HB_TYPE = ['prognosis/pretreatment_response'] 
SPECIFY_FOLD=5

STRATIFIED_CV_TOTAL_TRAININING_TIME = 20


IS_USING_WANDB = False

MAX_EPOCHS = 1000


# left_to_do SVM_ZHIFEI, RSFC_DUAN, NMF_ZHONG
MODELS_NEED_PREPROCESS_DATA = ['chao_cfnn',
                               'wang_alex', 
                            #    'zhu_xgboost', 
                               'yu_gnn',
                               'li_svm',
                               'decisiontree']

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
        'hb_path': 'hb_data.npy',
    },
    'gnn_transformer': {
        'hb_path': 'hb_data.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.001,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6, #np.random.choice([4, 8, 12]),
        'classweight1': 5,
    },
    'cnn_gnn_transformer': {
        'hb_path': 'hb_data.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.01,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6, #np.random.choice([4, 8, 12]),
        'classweight1': 5,
    },
    'gnn_transformer_tp_fc_fs': {
        'hb_path': 'merge_feature.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.01,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6,#np.random.choice([4, 8, 12]),
        'merge_sequence': [[0, 375],[375, 531],[531, 567],[567, 578]]
    },
    'gnn_transformer_tp_dp':{
        'hb_path': 'merge_feature.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.01,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6,#np.random.choice([4, 8, 12]),
        'merge_sequence': [[0, 375],[0,375],[567, 578]]
    },
    'mlp':{
        'hb_path': 'merge_psychiatry_demo_dose_data.npy',# 'merge_feature.npy', # hb_data
        'lr_v': 1e1,
        'l1_rate': 0.1,# should be 0.01
        'l2_rate': 0.1,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6, #np.random.choice([4, 8, 12]),
        'classweight1': 2,
    },
    'gnn_transformer_with_cli_demo':{
        'hb_path': 'hb_data.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.001,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6,#np.random.choice([4, 8, 12]),
        'cli_demo_shape': [19],
        'cli_dense_units': [32, 32, 16],
        'cli_demo_path': 'nor_demo_cli_merge_data.npy',
        'num_class': 2,
        'classweight1': 5,
    },
    'gnn_transformer_with_cli_demo_v1':{
        'hb_path': 'hb_data.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.001,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6,#np.random.choice([4, 8, 12]),
        'cli_demo_shape': [19],
        'cli_dense_units': [32, 32, 16],
        'cli_demo_path': 'nor_demo_cli_merge_data.npy',
        'num_class': 2,
        'classweight1': 25,
    },  
    'gnn_transformer_with_task_change_v1':{
        'hb_path': 'hb_data.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.001,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6,#np.random.choice([4, 8, 12]),
        'cli_demo_shape': [9],
        'cli_dense_units': [32, 32, 16],
        'cli_demo_path': 'task_change_hb_nine_regions_HbT.npy',
        'num_class': 2,
        'classweight1': 40,
    },        
    'gnn_transformer_with_task_change_v2':{
        'hb_path': 'hb_data.npy',# 'merge_feature.npy', # hb_data
        'adj_path': 'adj_matrix.npy',
        'l1_rate': 0.01,# should be 0.01
        'l2_rate': 0.001,# should be 0.001
        'd_model': 16,#np.random.choice([16, 32, 64]),
        'batch_size': 64,#np.random.choice([4, 8, 16, 64]),
        'n_layers': 6,#np.random.choice([4, 8, 12]),
        'cli_demo_shape': [9],
        'cli_dense_units': [32, 32, 16],
        'cli_demo_path': 'task_change_hb_nine_regions_HbT.npy',
        'num_class': 2,
        'last_transformer_dense_units': 1,
        'classweight1': 40,
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
        'hb_path': 'merge_psychiatry_demo_dose_data.npy',
    },
    'fusion_xgboost': {
        'hb_path': 'fusion_data.npy',
    },
    'fusion_catboost':{
        'hb_path': 'fusion_data.npy',
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
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
    },
    'decisiontree': {
        'hb_path': 'hb_data.npy',
    }
}
