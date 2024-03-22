from scripts.ML.Complete import TimeFeature_prognosis

from scripts.ML.development import DMFC_prognosis

print("Hello world!")
# while True:
#     TimeFeature_prognosis.start()
    
features_selection_methods = ['1_zhu', '2_zhifei', '3_wang', '4_yu']
hb_types = [ 'hbt', 'hbr', 'hbo']

models = ['XGBoost', 'Decision Tree'] ## ['Random Forest', 'SVM', 'KNN', 'Logistic Regression', 'Naive Bayes', 'Neural Network']
save_fold = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/results/ML_results/AutoML'
iteration_time = 5

while True:
    for model in models:
        for hb_type in hb_types:
            DMFC_prognosis.automl(model, hb_type, save_fold, iteration_time)
# DMFC_prognosis.start()

# while True:
#     for model in models:
#         for feature_selection_method in features_selection_methods:
#             TimeFeature_prognosis.automl(model, feature_selection_method, save_fold, iteration_time)
# TimeFeature_prognosis.start()