import pandas as pd

def preprocess_data(df):
    # Ensure the DataFrame is sorted properly
    df.sort_values(by=['conversation_ids', 'turn_ids', 'slice_ids'], inplace=True)
    
    # Create shifted columns for previous and next slice labels
    df['prev_label'] = df.groupby(['conversation_ids', 'turn_ids'])['true_label'].shift(1)
    df['next_label'] = df.groupby(['conversation_ids', 'turn_ids'])['true_label'].shift(-1)
    df['prev_prev_label'] = df.groupby(['conversation_ids', 'turn_ids'])['true_label'].shift(2)
    df['next_next_label'] = df.groupby(['conversation_ids', 'turn_ids'])['true_label'].shift(-2)

    return df

def calculate_custom_metrics(df):
    # Define conditions using vectorized operations
    condition_positive = (df['predicted_label'] == 1)
    condition_correct = (df['predicted_label'] == df['true_label'])

    condition_adjacent_match = (
        (df['prev_label'] == df['predicted_label']) |
        (df['next_label'] == df['predicted_label']) |
        (df['prev_prev_label'] == df['predicted_label']) |
        (df['next_next_label'] == df['predicted_label'])
    )

    # Calculate TP, FP, TN, FN using vectorized operations
    TP = ((condition_positive & condition_correct) | (condition_positive & condition_adjacent_match)).sum()
    TN = ((~condition_positive & condition_correct)).sum()
    FP = ((condition_positive & ~condition_correct) & ~condition_adjacent_match).sum()
    FN = ((~condition_positive & ~condition_correct)).sum()
    return TP, FP, TN, FN

def calculate_f1_score(TP, FP, TN, FN):
    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    return precision, recall, f1_score, accuracy
