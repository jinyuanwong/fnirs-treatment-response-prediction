def create_classifier(classifier_name, output_directory, callbacks, input_shape, epochs, info, sweep_config=None):
    if classifier_name == 'cnn_transformer':  # Time-CNN
        from classifiers import cnn_transformer
        return cnn_transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'pre_post_cnn_transformer':  # Time-CNN
        from classifiers import pre_post_cnn_transformer
        return pre_post_cnn_transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'transformer':  # Time-CNN
        from classifiers import transformer
        return transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gnn':  # Time-CNN
        from classifiers import gnn
        return gnn.Classifier_GNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gnn_transformer':  # Time-CNN
        from classifiers import gnn_transformer
        return gnn_transformer.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gnn_transformer_regression':  # Time-CNN
        from classifiers import gnn_transformer_regression
        return gnn_transformer_regression.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'mlp':  # Time-CNN
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'cnn_gnn_transformer':  # Time-CNN
        from classifiers import cnn_gnn_transformer
        return cnn_gnn_transformer.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gnn_transformer_with_cli_demo':  # Time-CNN
        from classifiers import gnn_transformer_with_cli_demo
        return gnn_transformer_with_cli_demo.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gnn_transformer_with_cli_demo_v1':  # Time-CNN
        from classifiers import gnn_transformer_with_cli_demo_v1
        return gnn_transformer_with_cli_demo_v1.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gnn_transformer_with_task_change_v1':  # Time-CNN
        from classifiers import gnn_transformer_with_task_change_v1
        return gnn_transformer_with_task_change_v1.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)    
    if classifier_name == 'gnn_transformer_with_task_change_v2':  # Time-CNN
        from classifiers import gnn_transformer_with_task_change_v2
        return gnn_transformer_with_task_change_v2.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)        
    if classifier_name == 'mamba':  # Time-CNN
        from classifiers import mamba
        return mamba.Classifier_Mamba(output_directory, callbacks, input_shape, epochs, sweep_config, info)    
    if classifier_name == 'jamba':  # Time-CNN
        from classifiers import jamba
        return jamba.Classifier_Jamba(output_directory, callbacks, input_shape, epochs, sweep_config, info)            
    if classifier_name == 'gin_transformer':  # Time-CNN
        from classifiers import gin_transformer
        return gin_transformer.Classifier_GIN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'graphformer':  # Time-CNN
        from classifiers import graphformer
        return graphformer.Classifier_Graph_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'rggcnn_transformer':  # Time-CNN
        from classifiers import rggcnn_transformer
        return rggcnn_transformer.Classifier_RGGCNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'mvg_transformer':  # Time-CNN
        from classifiers import mvg_transformer
        return mvg_transformer.Classifier_MVG_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'mgn_transformer':  # Time-CNN
        from classifiers import mvg_transformer
        return mvg_transformer.Classifier_MVG_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'mgm_transformer':  # Time-CNN
        from classifiers import mgm_transformer
        return mgm_transformer.Classifier_MGM_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'graphsage_transformer':  # Time-CNN
        from classifiers import graphsage_transformer
        return graphsage_transformer.Classifier_GraphSAGE_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'dgi_transformer':  # Time-CNN
        from classifiers import dgi_transformer
        return dgi_transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'yu_gnn':
        from classifiers import yu_gnn
        return yu_gnn.Classifier_GCN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'wang_alex':
        from classifiers import wang_alex
        return wang_alex.Classifier_AlexNet(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'comb_cnn':  # Time-CNN
        from classifiers import comb_cnn
        return comb_cnn.Classifier_CNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'chao_cfnn':  # Time-CNN
        from classifiers import chao_cfnn
        return chao_cfnn.Classifier_CFNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'zhu_xgboost':  # Time-CNN
        from classifiers import zhu_xgboost
        return zhu_xgboost.Classifier_XGBoost(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'fusion_xgboost':  # Time-CNN
        from classifiers import fusion_xgboost
        return fusion_xgboost.Classifier_XGBoost(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'fusion_catboost':  # Time-CNN
        from classifiers import fusion_catboost
        return fusion_catboost.Classifier_XGBoost(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'decisiontree':  # Time-CNN
        from classifiers import decisiontree
        return decisiontree.Classifier_DecisionTree(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    else:
        raise Exception(f'Your error message here, you did not register model {classifier_name} in classifier_factory.py!')
