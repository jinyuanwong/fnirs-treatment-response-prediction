import numpy as np
import os
from fine_tune_model import define_classifier_for_classification_for_response, define_classifier_for_classification_for_partial_response
from validation_method import stratified_5_fold_classification, nested_cross_validation_classification, loocv_classification
from utils_simple_prediction import add_task_change_data, load_data_for_classification, load_data_for_partial_response_prediction, add_cgi, add_mddr, set_path, load_task_change_data, save_model_seed, load_fnirs_feature_data
import time

def classification(data, labels, weight_0=None, partial_response=False):
    # Define the classifiers
    if partial_response is False:
        classifiers = define_classifier_for_classification_for_response(data, labels, weight_0)
    else: 
        classifiers = define_classifier_for_classification_for_partial_response(data, labels, weight_0)
    # loocv_classification(data, labels, classifiers)
    # stratified_5_fold_classification(data, labels, classifiers)
    inner_result, external_result = nested_cross_validation_classification(
        data, labels, classifiers)
    return inner_result, external_result


def choose_modality(modality, partial_response=False):

    data, labels = load_data_for_classification()
    save_fold_main = 'results/ML_results/simple_prediction'
    if partial_response:
        labels = load_data_for_partial_response_prediction()
        save_fold_main += '_partial_response'
    else:
        save_fold_main += '_response'
    if modality == 'clinical only':
        save_fold = save_fold_main + '/clinical_data_modality_classification/'
    if modality == 'fnirs only':
        save_fold = save_fold_main + '/fnirs_modality_classification/'
        data = load_task_change_data()[..., 2] # load_fnirs_feature_data(region_name='nine_regions') # load_task_change_data()[..., 2] # 2 -> HbT
    elif modality == 'clinical + fnirs':
        save_fold = save_fold_main + '/clinical_data_and_fnirs_modality_classification/'
        data = add_task_change_data(data)
        
    # data = add_mddr(data) 
            
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
    return data, labels, save_fold


def shuffle_input(data, labels, random_seed):

    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(data))
    shuffle_data = data[shuffled_indices]
    shuffle_labels = labels[shuffled_indices]
    
    # original_indices = [shuffled_indices.tolist().index(i) for i in range(data.shape[1])]
    return shuffle_data, shuffle_labels

def train(num_of_repeat, partial_response, random_seed = 42): 
    # weight_0 is used to tune class_weight or prior in the classifiers
    for weight_0 in [None]:#[i/10 for i in range (2, 10)]:#[None]: # np.arange(0.2, 1, 0.1): #[None]:# [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
        modalities =  ['clinical only', 'clinical + fnirs']  # ['fnirs only']  #  ['clinical only', 'fnirs only', 'clinical + fnirs']  # ['fnirs only']  #   ['clinical only', 'fnirs only', 'clinical + fnirs']  #['fnirs only'] #  ['clinical only', 'fnirs only', 'clinical + fnirs'] #   #      ['clinical only', 'clinical + fnirs']# 
        for i in range(num_of_repeat):
            for modality in modalities:
                random_seed += 1
                data, labels, save_fold = choose_modality(modality, partial_response=partial_response)

                shuffle_data, shuffle_labels = shuffle_input(data, labels, random_seed=random_seed)

                inner_result, external_result = classification(
                    shuffle_data, shuffle_labels, weight_0=weight_0, partial_response=partial_response)

                save_result = { # result[i] = 
                    'inner_result': inner_result,
                    'external_result': external_result
                }
                
                save_model_seed(save_fold, random_seed, save_result)

            # np.save(save_fold + 'ten_repeat_nested_cv.npy', result)


def evaluate_model(path):
    result = np.load(path, allow_pickle=True)
    result = result.item()
    TOTAL_REPETITION = len(result)
    # for key, value in result.items():
    #     print(key)
    #     print(value)
    #     print('--------------------------------------')
    classifiers_name = list(result[0]['external_result'].keys())
    metrics = list(result[0]['external_result'][classifiers_name[0]].keys())

    avg_inner_result = {classifier_name: {metric: []
                                          for metric in metrics} for classifier_name in classifiers_name}
    avg_external_result = {classifier_name: {metric: []
                                             for metric in metrics} for classifier_name in classifiers_name}

    for classifier in classifiers_name:
        for metric in metrics:

            avg_inner_result[classifier][metric] = [
                result[i]['inner_result'][classifier][metric] for i in range(TOTAL_REPETITION)]
            avg_external_result[classifier][metric] = [
                result[i]['external_result'][classifier][metric] for i in range(TOTAL_REPETITION)]

    print("\n## Inner Cross-Validation Performance")
    print("| Classifier | Average bAcc | Average Sensitivity | Average Specificity | Average F1 Score |")
    print("|------------|-----------------|------------------|---------------------|---------------------|")
    for name, metrics in avg_inner_result.items():
        avg_metrics = {metric: np.mean(scores)
                       for metric, scores in metrics.items()}
        print(
            f"| {name} | {avg_metrics['balanced accuracy']:.4f} | {avg_metrics['sensitivity']:.4f} | {avg_metrics['specificity']:.4f} | {avg_metrics['f1 score']:.4f} |")

    print("\n## Outer Cross-Validation Performance")
    print("| Classifier | Average bAcc | Average Sensitivity | Average Specificity | Average F1 Score |")
    print("|------------|-----------------|------------------|---------------------|---------------------|")
    for name, metrics in avg_external_result.items():
        avg_metrics = {metric: np.mean(scores)
                       for metric, scores in metrics.items()}

        print(
            f"| {name} | {avg_metrics['balanced accuracy']:.4f} | {avg_metrics['sensitivity']:.4f} | {avg_metrics['specificity']:.4f} | {avg_metrics['f1 score']:.4f} |")

def evaluate():
    
    for modality_name in ['clinical_data', 'fnirs' ,'clinical_data_and_fnirs']:
        path = f'results/ML_results/simple_prediction/{modality_name}_modality_classification/ten_repeat_nested_cv.npy'
        print(f"\n## {modality_name}")
        evaluate_model(path)
        print('\n--------------------------------------')    
if __name__ == "__main__":
    # change the working directory to the main folder
    set_path()
    time_seed = int(time.time())
    partial_response = False
    train(num_of_repeat=100, partial_response=partial_response, random_seed=time_seed) # random_seed=42

    # evaluate()
