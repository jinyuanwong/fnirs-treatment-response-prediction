import numpy as np 
import tensorflow as tf
import os 
import sys 

import shap


    
def compute_data_metrics(classifier, input_data, input_adj, y_true):
    # Assuming the TensorFlow  is stored in the variable `model`
    y_train_pred = classifier.model.predict([input_data, input_adj])
    y_train_pred_argmax = np.argmax(y_train_pred, axis=1)
    y_train_true = np.argmax(y_true, axis=1)
    from utils.utils_mine import get_metrics
    return get_metrics(y_train_pred_argmax, y_train_true)

main_pth = '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/'
sys.path.append(main_pth)
os.chdir(main_pth)
from LOO_nested_CV_model import build_model

def print_test_val_metrics(np_test_metrics, np_val_metrics):
    
    # np_test_metrics /= 100 
    # np_val_metrics /= 100
    
    mean_test_metrics = np.mean(np_test_metrics, axis=0)
    std_test_metrics = np.std(np_test_metrics, axis=0)

    mean_val_metrics = np.mean(np_val_metrics, axis=0)
    std_val_metrics = np.std(np_val_metrics, axis=0)
    
    from utils.fnirs_utils import print_md_table_val_test_AUC
    print_md_table_val_test_AUC('Mean', mean_test_metrics, mean_val_metrics, already_balanced_accuracy=False)
    print_md_table_val_test_AUC('STD', std_test_metrics, std_val_metrics, already_balanced_accuracy=False, print_table_header=False)
    
def save(test_metrics, val_metrics, y_pred_test_all, save_fold):

    np_test_metrics = np.array(test_metrics)
    np_test_metrics[:, 0] = (np_test_metrics[:, 1] + np_test_metrics[:, 0]) / 2
    print(np_test_metrics)
    np_val_metrics = np.array(val_metrics)
    np_val_metrics = np.reshape(np_val_metrics, (-1, 64, 5, 4))
    np_val_metrics = np.mean(np_val_metrics, axis=(1,2))
    np_val_metrics[:, 0] = (np_val_metrics[:, 1] + np_val_metrics[:, 2])/2
    np_y_pred_test_all = np.array(y_pred_test_all)


    print_test_val_metrics(np_test_metrics, np_val_metrics)

    # 
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
    np.save(save_fold + '/val_metrics.npy', np_val_metrics)
    np.save(save_fold + '/test_metrics.npy', np_test_metrics)
    np.save(save_fold + '/y_pred_test_all.npy', np_y_pred_test_all)    

model_name =  'mamba' # 'gnn_transformer_with_task_change_v2' 'gnn_transformer_with_cli_demo_v1'#'gnn_transformer_with_cli_demo' # 'gnn_transformer' gnn_transformer_with_task_change_v1
config_file_name = 'pretreatment_response_mamba' # pretreatment_response_cv_5_mix_hb pretreatment_response_mamba
msg = 'loocv_v4to13'
TrainModel = build_model(model_name, config_file_name, msg)

from utils.utils_mine import stratified_LOO_nested_CV

data = TrainModel.data
label = TrainModel.label
# adj = TrainModel.adj
# if TrainModel.cli_demo.any():
#     cli_demo = TrainModel.cli_demo
# else:
#     cli_demo = None
    
current_loo = 0
k = 0
num_of_k_fold = 5
classifier = TrainModel.model
# X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test = stratified_LOO_nested_CV(data, label, k=k, num_of_k_fold=num_of_k_fold, current_loo=current_loo, adj=adj)

X_train, Y_train, X_val, Y_val, X_test, Y_test = stratified_LOO_nested_CV(data, label, k=k, num_of_k_fold=num_of_k_fold, current_loo=current_loo)

# print(f'X_train: {X_train.shape}')
# print(f'X_val: {X_val.shape}')
# print(f'X_test: {X_test.shape}')


from utils.fnirs_utils import get_metrics_auc

all_subjects_shap = []
val_metrics = []
test_metrics = []
y_pred_test_all = []
y_pred_val_all = []
# for v_i in range(6):
y_test_prediction_prob = []
model_name = 'mamba'
for v_i in range(1):
    
    model_para = f'loocv_v{v_i}' + config_file_name
    y_pred_test_fold = []
    for subject in range(64):
        for k_fold in range(num_of_k_fold):
            X_train, Y_train, X_val, Y_val, X_test, Y_test = stratified_LOO_nested_CV(data, label, k=k_fold, num_of_k_fold=num_of_k_fold, current_loo=subject)
            checkpoint_path = f'results/{model_name}/prognosis_mix_hb/pretreatment_response/{model_para}/LOO_nested_CV/LOO_{subject}/stratified_nested_5_CV_fold-{k_fold}/fold-best-checkpoint' #checkpoint' # fold-best-checkpoint

            classifier.model.load_weights(checkpoint_path)
            model = classifier.model
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            # y_test_prediction_prob.append(y_pred_test)
            y_test_prediction_prob.append(y_pred_test[0, 1])
            y_pred_test_fold.append(y_pred_test)
            val_metrics.append(get_metrics_auc(Y_val.argmax(axis=1), y_pred_val))
    y_pred_test_fold = np.array(y_pred_test_fold)
    y_pred_test_fold = np.squeeze(y_pred_test_fold)
    y_pred_test_fold = y_pred_test_fold.reshape(64, 5, 2)
    y_pred_test_fold = np.mean(y_pred_test_fold, axis=1)
    y_pred_test_all.append(y_pred_test_fold)
    test_metrics.append(get_metrics_auc(label.argmax(axis=1), y_pred_test_fold))
            # NOTE TO DO: why the train metrics are very bad? see sub-0 fold-0/1 
            # train_metrics = compute_data_metrics(classifier, X_train, adj_train, Y_train)
            # val_metrics = compute_data_metrics(classifier, X_val, adj_val, Y_val)
            # print(f"sub - {subject} - fold - {k_fold} - train metrics - {train_metrics}")
            # print(f"sub - {subject} - fold - {k_fold} - val metrics - {val_metrics}")
            
            # explainer = shap.GradientExplainer(classifier.model, [X_train, adj_train])
            # shap_values, _ = explainer.shap_values([X_test, adj_test])
            # all_subjects_shap.append(shap_values)

save_fold = f'results/{model_name}/prognosis_mix_hb/pretreatment_response/ten-fold-evaluate'
save(test_metrics, val_metrics, y_pred_test_all, save_fold)


"""
using checkpoint 2024-06-31

| Model Name | Testing Set |             |             |             | Validation Set |             |             |             |
|------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
|            | Balanced Accuracy | Sensitivity | Specificity | AUC | Balanced Accuracy | Sensitivity | Specificity | AUC |
| Mean   | 62.3810  | 53.3333  | 71.4286  | 70.4762  | 71.0243  | 62.6042  | 79.4444  | 68.3275  |
| STD   | 0.0000  | 0.0000  | 0.0000  | 0.0000  | 0.0000  | 0.0000  | 0.0000  | 0.0000  |

"""