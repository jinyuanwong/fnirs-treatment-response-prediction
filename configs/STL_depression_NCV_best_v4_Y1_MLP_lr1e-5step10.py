from configs.config import *
from configs.models_args.transformer_args import Transformer_ModelArgs
import os
import tensorflow_addons as tfa
from classifiers.loss.focal_loss import focal_loss
import tensorflow.keras as keras
# Define the loss dictionary using Seesaw Loss from tensorflow_addons
# loss = {
#     'gender': tf.keras.losses.CategoricalCrossentropy(),
#     'smoking': tfa.losses.SeesawLoss(),
#     'alcohol': tfa.losses.SeesawLoss(),
#     'Suicide_Risk': tfa.losses.SeesawLoss(),
#     'depression': tfa.losses.SeesawLoss()
# }
from utils.callbacks import reduceLRonplateau
from configs.mdd_classification_jamba import *
INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 4
OUTER_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
HOLD_OUT_DIV = 5
# MONITOR_METRIC = 'depression_accuracy'
BEGIN_PATIENCE = 11 # compared from 0 to 30
AUGMENT_RATIO = 0
MIN_DELETE_CHANNEL = 8
MAX_DELETE_CHANNEL = 17
# AUGMENT_RATIO = 20
focal_loss_fn = keras.losses.CategoricalCrossentropy()


transformer_args = Transformer_ModelArgs(
    batch_size=64,
    d_model=64,
    n_heads=1,
    class_weights={0: 1, 1: 1},  # for pretreament classification {0: 1, 1: 5}
)


PARAMETER['cnn_transformer'] = {
    'args': transformer_args,
    'config_file_path': [os.path.abspath(__file__)],
}

args = Jamba_ModelArgs_extend_from_Mamba(
    monitor_metric_mode = 'min',
    monitor_metric_checkpoint = 'val_loss',
    monitor_metric_early_stop = 'val_accuracy',
    load_previous_checkpoint = True,
    batch_size=64,
    classweight1=1,
    patiences=calculate_patience(AUGMENT_RATIO, begin_patience=BEGIN_PATIENCE),
    lr_mode='CosineDecayRestarts',
    lr_begin=1e-5, # compared from 1e-3 to 1e-7
    lr_first_decay_steps=10,
    model_input_dims=512,
    model_states=128,  # No mamba this will not be used
    last_dense_units=64,
    num_layers= 0,  # compared from 0 to 10
    dropout_rate=0.4,  # compared from 0.1 to 0.7
    vocab_size=2,
    num_classes=2,
    warmup_step=100,
    delete_checkpoint = True,
    reduce_lr = None,

    # loss={'gender': 'categorical_crossentropy', 'smoking':  tfa.losses.SeesawLoss(), 'alcohol':  tfa.losses.SeesawLoss(),
    #       'Suicide_Risk':  tfa.losses.SeesawLoss(), 'depression': 'categorical_crossentropy'},  # 'binary_crossentropy', # categorical_crossentropy
    loss={
          'depression': focal_loss_fn, #'categorical_crossentropy',
          }, #'categorical_crossentropy'},  # 'binary_crossentropy', # categorical_crossentropy    
    metrics={
            'depression': 'accuracy',
            },
    
    projection_expand_factor=1,
    clipnorm = 1.0,
    weight_decay= 0.004,
    use_mlp_layer = True, #
    use_gnn_layer = False, #
    use_conv1d_layer = False, #
    use_mamba_block = False, #
)

PARAMETER['jamba_MTL_V2'] = PARAMETER['jamba_MTL'] = {
    'args': args,
    'config_file_path': [os.path.abspath(__file__)],
}


for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_simple_all_1d.npy'
    PARAMETER[model]['label_path'] = 'multi_task_label_depression_onehot.npy'
    PARAMETER[model]['classweight1'] = 1