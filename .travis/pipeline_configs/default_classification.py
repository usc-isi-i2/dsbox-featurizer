class config:
    config = {'sampling_step': {'primitive': 'd3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX', 'hyperparameters': {}}, 'denormalize_step': {'primitive': 'd3m.primitives.data_transformation.denormalize.Common', 'hyperparameters': {}}, 'to_dataframe_step': {'primitive': 'd3m.primitives.data_transformation.dataset_to_dataframe.Common', 'hyperparameters': {}}, 'extract_attribute_step': {'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon', 'hyperparameters': {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'https://metadata.datadrivendiscovery.org/types/Attribute')}}, 'profiler_step': {'primitive': 'd3m.primitives.schema_discovery.profiler.DSBOX', 'hyperparameters': {}}, 'clean_step': {'primitive': 'd3m.primitives.data_cleaning.cleaning_featurizer.DSBOX', 'hyperparameters': {}}, 'encode_step': {'primitive': 'd3m.primitives.data_preprocessing.encoder.DSBOX', 'hyperparameters': {}}, 'corex_step': {'primitive': 'd3m.primitives.feature_construction.corex_text.CorexText', 'hyperparameters': {}}, 'to_numeric_step': {'primitive': 'd3m.primitives.data_transformation.to_numeric.DSBOX', 'hyperparameters': {}}, 'impute_step': {'primitive': 'd3m.primitives.data_preprocessing.mean_imputation.DSBOX', 'hyperparameters': {}}, 'scaler_step': {'primitive': 'd3m.primitives.data_preprocessing.do_nothing.DSBOX', 'hyperparameters': {}}, 'data': {'primitive': 'd3m.primitives.data_preprocessing.do_nothing.DSBOX', 'hyperparameters': {}}, 'pre_target': {'primitive': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon', 'hyperparameters': {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',)}}, 'target': {'primitive': 'd3m.primitives.data_transformation.to_numeric.DSBOX', 'hyperparameters': {'drop_non_numeric_columns': False}}, 'feature_selector_step': {'primitive': 'd3m.primitives.feature_selection.generic_univariate_select.SKlearn', 'hyperparameters': {'use_semantic_types': True, 'return_result': 'new', 'add_index_columns': True, 'mode': 'percentile', 'param': 10}}, 'model_step': {'primitive': 'd3m.primitives.classification.random_forest.SKlearn', 'hyperparameters': {'use_semantic_types': True, 'return_result': 'new', 'add_index_columns': True, 'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 5, 'max_features': 'sqrt', 'n_estimators': 100}}}
    pipeline_type = "classification"
    test_dataset_id = "38_sick"