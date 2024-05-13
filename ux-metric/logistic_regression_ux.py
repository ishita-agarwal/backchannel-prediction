import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
# from prediction_analysis import custom_accuracy_metric, calculate_f1_score
from prediction_analysis_vectorized import preprocess_data, calculate_custom_metrics, calculate_f1_score
with open('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/classifier_input_ux/feature_matrix.pkl', 'rb') as f:
    features = pickle.load(f)

# Separate 'conversation_ids' and 'label' from the features
X = features.drop(['label', 'conversation_ids'], axis=1)
y = features['label']
conversation_ids = features['conversation_ids']
slice_ids = features['slice_ids']
turn_ids = features['turn_ids']
# Setup the splitter
split = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

# Use 'conversation_id' for creating groups to split, not as a feature
for train_idx, test_idx in split.split(X, y, groups=conversation_ids):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    test_conversation_ids, turn_ids_test, slice_ids_test = conversation_ids.iloc[test_idx], turn_ids.iloc[test_idx], slice_ids.iloc[test_idx]  # Preserve test set conversation_ids


tolerance = 1e-2
clf = LogisticRegression(penalty='l2', tol=tolerance)  
clf.fit(X_train, y_train)

# Predict probabilities for test set
y_proba = clf.predict_proba(X_test)[:, 1]

# Function to adjust threshold
def adjust_threshold(y_proba, threshold):
    return [1 if prob >= threshold else 0 for prob in y_proba]

# Adjust threshold
threshold = 0.07  # Set your desired threshold here
y_pred_adjusted = adjust_threshold(y_proba, threshold)

# Reset indices explicitly
test_conversation_ids_reset = test_conversation_ids.reset_index(drop=True)
turn_ids_test_reset = turn_ids_test.reset_index(drop=True)
slice_ids_test_reset = slice_ids_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
y_pred_adjusted_series = pd.Series(y_pred_adjusted).reset_index(drop=True)

# Create DataFrame with these new series
predictions_df = pd.DataFrame({
    'conversation_ids': test_conversation_ids_reset,
    'turn_ids': turn_ids_test_reset,
    'slice_ids': slice_ids_test_reset,
    'predicted_label': y_pred_adjusted_series,
    'true_label': y_test_reset
})

predictions_df.to_pickle('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/prediction_ux/predictions_with_ids.pkl')
predictions_df = preprocess_data(predictions_df)
# Evaluate classification report

print("X_test shape", X_test.shape)
print("y_pred_adjusted shape", len(y_pred_adjusted))
TP, FP, TN, FN = calculate_custom_metrics(predictions_df)
precision, recall, f1, custom_accuracy = calculate_f1_score(TP, FP, TN, FN)
print("Custom Accuracy:", custom_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

lr_auc = roc_auc_score(y_test, y_proba)
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)