Order to run -
1. get_features_final (to generate all features, excluding embeddings)
2. pre_classifier (to generate feature , label matrix for classifier input)
3. logistic_regression_new (classifier)

get_features_new adds extra features for conversation_id, turn_id and slice_id - used for user experience metric