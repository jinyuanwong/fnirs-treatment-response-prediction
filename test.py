from scripts.ML.Complete import TimeFeature_prognosis
# features_selection_methods = ['1_zhu', '2_zhifei', '3_wang', '4_yu']
# hb_types = [ 'hbt', 'hbr', 'hbo']
# models = ['XGBoost', 'Decision Tree'] ## ['Random Forest', 'SVM', 'KNN', 'Logistic Regression', 'Naive Bayes', 'Neural Network']
# save_fold = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/results/ML_results/AutoML'
# iteration_time = 5

# dataset = 'hb_data_v1' # or hb_data note: hb_data_v1 is hb_data processed with 6_mannully_delete_signal
_, _, _, model = TimeFeature_prognosis.predict_based_on_automl(using_CV=True)#(csv_pth, dataset)
TimeFeature_prognosis.plot_model_importance(model)