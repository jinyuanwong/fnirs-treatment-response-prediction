from configs.config import *
MAX_EPOCHS = 500


INPUT_HB_TYPE = ['prognosis_mix_hb/pretreatment_response'] 
for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_data.npy'
    PARAMETER[model]['classweight1'] = 5