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
OUTER_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
HOLD_OUT_DIV = 5
# MONITOR_METRIC = 'depression_accuracy'
AUGMENT_RATIO = 0
MIN_DELETE_CHANNEL = 8
MAX_DELETE_CHANNEL = 17
# AUGMENT_RATIO = 20
focal_loss_fn = focal_loss(alpha=0.25, gamma=2.0) # V2 

transformer_args = Transformer_ModelArgs(
    batch_size=64,
    d_model=16,
    n_heads=1,
    class_weights={0: 1, 1: 1},  # for pretreament classification {0: 1, 1: 5}
    num_classes=2,
    warmup_step=4000,
    loss={
          'depression': focal_loss_fn, #'categorical_crossentropy',
          }, #'categorical_crossentropy'},  # 'binary_crossentropy', # categorical_crossentropy    
    metrics={
            'depression': 'accuracy',
            },
)


PARAMETER['cnn_transformer'] = {
    'args': transformer_args,
    'config_file_path': [os.path.abspath(__file__)],
}

for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_simple_all_1d.npy'
    PARAMETER[model]['label_path'] = 'multi_task_label_depression_onehot.npy'
    PARAMETER[model]['classweight1'] = 1
