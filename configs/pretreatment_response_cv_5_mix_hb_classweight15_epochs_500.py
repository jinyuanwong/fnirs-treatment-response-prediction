from configs.config import *


INPUT_HB_TYPE = ['prognosis_mix_hb/pretreatment_response']
SPECIFY_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000



parameters_cv_5_mix_hb = {
        'epochs': 500,
        'classweight1': 15,
    }

for key, value in parameters_cv_5_mix_hb.items():
    PARAMETER['gnn_transformer'][key] = value
    PARAMETER['cnn_gnn_transformer'][key] = value
    PARAMETER['gnn_transformer_with_cli_demo'][key] = value