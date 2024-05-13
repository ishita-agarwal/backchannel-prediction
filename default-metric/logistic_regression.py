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
from sklearn.model_selection import KFold, GroupShuffleSplit
from sklearn.model_selection import cross_val_score

with open('/final/src/data/classifier_input_ux/feature_matrix.pkl', 'rb') as f:
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

# For k-fold cross validation
# k = 5
# kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Train logistic regression model
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

# Evaluate classification report
# print(classification_report(y_test, y_pred_adjusted))
print("X_test shape", X_test.shape)
print("y_pred_adjusted shape", len(y_pred_adjusted))
# cross_validation_score = cross_val_score(clf, X, y, cv=kf, scoring='f1')
accuracy = accuracy_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# print("Cross Validation Score:", cross_validation_score)
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_adjusted, labels=clf.classes_)
# Print confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

lr_auc = roc_auc_score(y_test, y_proba)
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()