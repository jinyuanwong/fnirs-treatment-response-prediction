def create_classifier(classifier_name, output_directory, callbacks, input_shape, epochs, info, sweep_config=None):
    if classifier_name == 'cnn_transformer':  # Time-CNN
        import cnn_transformer
        return cnn_transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'pre_post_cnn_transformer':  # Time-CNN
        import pre_post_cnn_transformer
        return pre_post_cnn_transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'transformer':  # Time-CNN
        import transformer
        return transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gnn':  # Time-CNN
        import gnn
        return gnn.Classifier_GNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gnn_transformer':  # Time-CNN
        import gnn_transformer
        return gnn_transformer.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'gin_transformer':  # Time-CNN
        import gin_transformer
        return gin_transformer.Classifier_GIN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'graphformer':  # Time-CNN
        import graphformer
        return graphformer.Classifier_Graph_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'rggcnn_transformer':  # Time-CNN
        import rggcnn_transformer
        return rggcnn_transformer.Classifier_RGGCNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'mvg_transformer':  # Time-CNN
        import mvg_transformer
        return mvg_transformer.Classifier_MVG_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'mgn_transformer':  # Time-CNN
        import mvg_transformer
        return mvg_transformer.Classifier_MVG_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'mgm_transformer':  # Time-CNN
        import mgm_transformer
        return mgm_transformer.Classifier_MGM_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'graphsage_transformer':  # Time-CNN
        import graphsage_transformer
        return graphsage_transformer.Classifier_GraphSAGE_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'dgi_transformer':  # Time-CNN
        import dgi_transformer
        return dgi_transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'yu_gnn':
        import yu_gnn
        return yu_gnn.Classifier_GCN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'wang_alex':
        import wang_alex
        return wang_alex.Classifier_AlexNet(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'comb_cnn':  # Time-CNN
        import comb_cnn
        return comb_cnn.Classifier_CNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'chao_cfnn':  # Time-CNN
        import chao_cfnn
        return chao_cfnn.Classifier_CFNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    if classifier_name == 'zhu_xgboost':  # Time-CNN
        import zhu_xgboost
        return zhu_xgboost.Classifier_XGBoost(output_directory, callbacks, input_shape, epochs, sweep_config, info)
    else:
        raise Exception('Your error message here')
