from configs.config import *
from configs.models_args.transformer_args import Transformer_ModelArgs
import os
import tensorflow_addons as tfa
from classifiers.loss.focal_loss import focal_loss
# Define the loss dictionary using Seesaw Loss from tensorflow_addons
# loss = {
#     'gender': tf.keras.losses.CategoricalCrossentropy(),
#     'smoking': tfa.losses.SeesawLoss(),
#     'alcohol': tfa.losses.SeesawLoss(),
#     'Suicide_Risk': tfa.losses.SeesawLoss(),
#     'depression': tfa.losses.SeesawLoss()
# }
from configs.mdd_classification_jamba import *
INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 4
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 100
HOLD_OUT_DIV = 5
MONITOR_METRIC = 'depression_accuracy'
AUGMENT_RATIO = 0
focal_loss_fn = focal_loss(alpha=0.25, gamma=2.0) # V2 

transformer_args = Transformer_ModelArgs(
    batch_size=64,
    d_model=64,
    n_heads=1,
    class_weights={0: 1, 1: 1},  # for pretreament classification {0: 1, 1: 5}
)


PARAMETER['cnn_transformer'] = {
    'hb_path': 'nor_hb_simple_all_1d.npy',
    'args': transformer_args,
}

args = Jamba_ModelArgs_extend_from_Mamba(
    monitor_metric_mode = 'max',
    monitor_metric_checkpoint = 'val_depression_accuracy',
    load_previous_checkpoint = True,
    batch_size=32,
    classweight1=1,
    patiences=5,
    lr_begin=1e7,  # 1e7 -> 1e5
    model_input_dims=128,
    model_states=64,  # 64 -> 128
    last_dense_units=64,
    num_layers= 5,  # 2 -> 1
    dropout_rate=0.3,  # 0.35 -> 0.15
    vocab_size=2,
    num_classes=2,
    warmup_step=4000,
    # loss={'gender': 'categorical_crossentropy', 'smoking':  tfa.losses.SeesawLoss(), 'alcohol':  tfa.losses.SeesawLoss(),
    #       'Suicide_Risk':  tfa.losses.SeesawLoss(), 'depression': 'categorical_crossentropy'},  # 'binary_crossentropy', # categorical_crossentropy
    loss={
          'gender': focal_loss_fn, #'categorical_crossentropy', 
          'age': focal_loss_fn, #'categorical_crossentropy',
          'education': focal_loss_fn, #'categorical_crossentropy', 
          'smoking': focal_loss_fn, #'categorical_crossentropy', 
          'alcohol': focal_loss_fn, #'categorical_crossentropy',
          'HAMD_Scores': focal_loss_fn, #'categorical_crossentropy',
          'Suicide_Risk': focal_loss_fn, #'categorical_crossentropy', 
          'depression': focal_loss_fn}, #'categorical_crossentropy'},  # 'binary_crossentropy', # categorical_crossentropy    
    metrics={
            'gender': 'accuracy', 
            'age': 'accuracy', 
            'education': 'accuracy', 
            'smoking': 'accuracy', 
            'alcohol': 'accuracy',
            'HAMD_Scores': 'accuracy',
            'Suicide_Risk': 'accuracy', 
            'depression': 'accuracy'},
    projection_expand_factor=1,
)

PARAMETER['mamba_MTL'] = PARAMETER['jamba_MTL'] = {
    'hb_path': 'hbo_simple_data.npy',
    'args': args,
    'config_file_path': [os.path.abspath(__file__)],
}


for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_seq_ch_hb_simple_all_1d.npy' # nor_seq_ch_hb_data_1d nor_seq_ch_hb_simple_all_1d 
    PARAMETER[model]['label_path'] = 'multi_task_label_5_one_hot_encoded.npy'
    PARAMETER[model]['classweight1'] = 1
