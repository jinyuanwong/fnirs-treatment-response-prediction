from configs.config import *


INPUT_HB_TYPE = ['prognosis/pretreatment_response'] 
for model, val in PARAMETER.items():
    # PARAMETER[model]['hb_path'] = 'hbo_simple_data.npy'
    PARAMETER[model]['classweight1'] = 5