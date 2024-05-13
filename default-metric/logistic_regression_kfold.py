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
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []

# Setup k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


for train_index, test_index in kf.split(X, groups=conversation_ids):  # Ensure groups are not split
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train model
    clf = LogisticRegression(penalty='l2', tol=1e-2)
    clf.fit(X_train, y_train)
    # Predict probabilities for test set
    y_proba = clf.predict_proba(X_test)[:, 1]
    # Adjust threshold
    threshold = 0.07  # Set your desired threshold here
    y_pred_adjusted = adjust_threshold(y_proba, threshold)

    # Store results
    accuracy_scores.append(accuracy_score(y_test, y_pred_adjusted))
    f1_scores.append(f1_score(y_test, y_pred_adjusted))
    precision_scores.append(precision_score(y_test, y_pred_adjusted))
    recall_scores.append(recall_score(y_test, y_pred_adjusted))
# Output average results over all folds
print("All F1 scores", f1_scores)
print("All accuracy scores", accuracy_scores)
print("All Precision scores", precision_scores)
print("All Recall scores", recall_scores)

print("Average F1 Score:", np.mean(f1_scores))
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision score", np.mean(precision_scores))
print("Average Recall score", np.mean(recall_scores))