import numpy as np
import os
from fine_tune_model import define_classifier_for_classification
from validation_method import stratified_5_fold_classification, nested_cross_validation_classification, loocv_classification
from utils_simple_prediction import add_task_change_data, load_data_for_classification, add_cgi, add_mddr, set_path, load_task_change_data


def classification(data, labels):
    # Define the classifiers
    classifiers = define_classifier_for_classification(data, labels)
    # loocv_classification(data, labels, classifiers)
    # stratified_5_fold_classification(data, labels, classifiers)
    inner_result, external_result = nested_cross_validation_classification(
        data, labels, classifiers)
    return inner_result, external_result


def choose_modality(modality):

    data, labels = load_data_for_classification()
    if modality == 'clinical only':
        save_fold = 'results/ML_results/simple_prediction/clinical_data_modality_classification/'
    if modality == 'fnirs only':
        save_fold = 'results/ML_results/simple_prediction/fnirs_modality_classification/'
        data = load_task_change_data()[..., 2] # 2 -> HbT
    elif modality == 'clinical + fnirs':
        save_fold = 'results/ML_results/simple_prediction/clinical_data_and_fnirs_modality_classification/'
        data = add_task_change_data(data)
    elif modality == 'clinical + MDDR':
        save_fold = 'results/ML_results/simple_prediction/clinical_data_and_mddr_modality_classification/'
        data = add_mddr(data)
    elif modality == 'clinical + fnirs + MDDR':
        save_fold = 'results/ML_results/simple_prediction/clinical_data_and_fnirs_mddr_modality_classification/'
        data = add_task_change_data(data)
        data = add_mddr(data)

    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
    return data, labels, save_fold


def train(num_of_repeat, random_seed = 42):

    modalities = ['fnirs only']#['clinical + fnirs',
                  #'clinical + MDDR', 'clinical + fnirs + MDDR']
    for modality in modalities:
        data, labels, save_fold = choose_modality(modality)

        # Shuffle the data and labels
        
        result = {}
        # num_of_repeat = 1
        for i in range(num_of_repeat):
            random_seed += i
            np.random.seed(random_seed)
            shuffled_indices = np.random.permutation(len(data))
            shuffle_data = data[shuffled_indices]
            shuffle_labels = labels[shuffled_indices]

            inner_result, external_result = classification(
                shuffle_data, shuffle_labels)

            result[i] = {
                'inner_result': inner_result,
                'external_result': external_result
            }

        np.save(save_fold + 'ten_repeat_nested_cv.npy', result)


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

    train(num_of_repeat=5, random_seed=1024) # random_seed=42

    evaluate()
