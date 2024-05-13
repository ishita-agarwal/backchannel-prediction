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
from prediction_analysis_vectorized import preprocess_data, calculate_custom_metrics, calculate_f1_score
with open('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/classifier_input_ux/feature_matrix.pkl', 'rb') as f:
    features = pickle.load(f)

# Function to adjust threshold
def adjust_threshold(y_proba, threshold):
    return [1 if prob >= threshold else 0 for prob in y_proba]


# Separate 'conversation_ids' and 'label' from the features
X = features.drop(['label', 'conversation_ids'], axis=1)
y = features['label']
conversation_ids = features['conversation_ids']
slice_ids = features['slice_ids']
turn_ids = features['turn_ids']


# Setup k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize variables to collect metrics
custom_accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
auc_scores = []

for train_index, test_index in kf.split(X, groups=conversation_ids):  # Ensure groups are not split
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    conversation_ids_train, conversation_ids_test = conversation_ids.iloc[train_index], conversation_ids.iloc[test_index]
    turn_ids_train, turn_ids_test = turn_ids.iloc[train_index], turn_ids.iloc[test_index]
    slice_ids_train, slice_ids_test = slice_ids.iloc[train_index], slice_ids.iloc[test_index]

    # Train model
    clf = LogisticRegression(penalty='l2', tol=1e-2)
    clf.fit(X_train, y_train)
    # Predict probabilities for test set
    y_proba = clf.predict_proba(X_test)[:, 1]
    # Adjust threshold
    threshold = 0.07  # Set your desired threshold here
    y_pred_adjusted = adjust_threshold(y_proba, threshold)
    # Predict
    
    # Reset indices explicitly
    test_conversation_ids_reset = conversation_ids_test.reset_index(drop=True)
    turn_ids_test_reset = turn_ids_test.reset_index(drop=True)
    slice_ids_test_reset = slice_ids_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    y_pred_adjusted_series = pd.Series(y_pred_adjusted).reset_index(drop=True)

    # Create DataFrame for custom metric evaluation
    predictions_df = pd.DataFrame({
        'conversation_ids': test_conversation_ids_reset,
        'turn_ids': turn_ids_test_reset,
        'slice_ids': slice_ids_test_reset,
        'predicted_label': y_pred_adjusted_series,
        'true_label': y_test_reset
    })

    # Calculate custom metrics
    print("Fold")
    predictions_df = preprocess_data(predictions_df)
    TP, FP, TN, FN = calculate_custom_metrics(predictions_df)
    precision, recall, f1, custom_accuracy = calculate_f1_score(TP, FP, TN, FN)
    lr_auc = roc_auc_score(y_test, y_proba)
    # Store results
    custom_accuracy_scores.append(custom_accuracy)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    auc_scores.append(lr_auc)

# Output average results over all folds
print("All F1 scores", f1_scores)
print("All Custom Accuracy scores", custom_accuracy_scores)
print("All Precision scores", precision_scores)
print("All Recall scores", recall_scores)
print("AUC scores", auc_scores)

print("Average F1 score:", np.mean(f1_scores))
print("Average Custom Accuracy score:", np.mean(custom_accuracy_scores))
print("Average Precision score", np.mean(precision_scores))
print("Average Recall score", np.mean(recall_scores))
print("Average AUC score", np.mean(auc_scores))