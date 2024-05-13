import pickle

def custom_accuracy_metric(df):
    # Initialize counters
    correct_predictions = 0
    applicable_cases = 0
    TP = FP = TN = FN = 0
    # Iterate through DataFrame rows
    for idx, row in df.iterrows():
        # print("here line 10")
        applicable_cases += 1
        # Define the current slice, turn, and conversation IDs
        current_slice_id = row['slice_ids']
        current_turn_id = row['turn_ids']
        current_conversation_id = row['conversation_ids']
        current_predicted_label = row['predicted_label']
        current_true_label = row['true_label']

        if current_predicted_label == current_true_label:  # default case
            # print("Here")
            if current_predicted_label == 1:  # Predicted positive and it's true
                TP += 1
            else:  # Predicted negative and it's true
                TN += 1
            correct_predictions += 1
        elif current_predicted_label == 1:
            
            previous_previous_slice = df[(df['slice_ids'] == current_slice_id - 2) &
                                (df['turn_ids'] == current_turn_id) &
                                (df['conversation_ids'] == current_conversation_id)]
            
            # Find the previous slice in the same turn and conversation
            previous_slice = df[(df['slice_ids'] == current_slice_id - 1) &
                                (df['turn_ids'] == current_turn_id) &
                                (df['conversation_ids'] == current_conversation_id)]

            # Find the next slice in the same turn and conversation
            next_slice = df[(df['slice_ids'] == current_slice_id + 1) &
                            (df['turn_ids'] == current_turn_id) &
                            (df['conversation_ids'] == current_conversation_id)]

            # Find the next slice in the same turn and conversation
            next_next_slice = df[(df['slice_ids'] == current_slice_id + 2) &
                            (df['turn_ids'] == current_turn_id) &
                            (df['conversation_ids'] == current_conversation_id)]
            
            # Check conditions for previous and next slices
            if not previous_previous_slice.empty:
                previous_previous_condition = previous_previous_slice['true_label'].values[0] == current_predicted_label
            else:
                previous_previous_condition = False

            if not previous_slice.empty:
                previous_condition = previous_slice['true_label'].values[0] == current_predicted_label
            else:
                previous_condition = False

            if not next_slice.empty:
                next_condition = next_slice['true_label'].values[0] == current_predicted_label
            else:
                next_condition = False

            if not next_next_slice.empty:
                next_next_condition = next_next_slice['true_label'].values[0] == current_predicted_label
            else:
                next_next_condition = False
            
            # If either previous or next slice true label matches the current slice's prediction, increase counts
            if previous_previous_condition or previous_condition or next_condition or next_next_condition:
                correct_predictions += 1
                TP += 1
            else:  # Predicted positive but it's false
                FP += 1
        else:  # Predicted negative but it's false
            FN += 1

    # Calculate custom metric: accuracy of correctly labeled as per custom rule
    if applicable_cases > 0:
        return (TP, FP, TN, FN, (correct_predictions / applicable_cases))
    else:
        return None  # or another appropriate value if no applicable cases are found

def calculate_f1_score(TP, FP, TN, FN):
    # Calculate precision and recall
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

# with open('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/prediction_ux/predictions_with_ids.pkl', 'rb') as f:
#     predictions_df = pickle.load(f)
# # print(predictions_df['conversation_ids'].unique())
# # predictions_df = predictions_df[predictions_df['conversation_ids'] == 16]

# # Calculate the custom metric
# TP, FP, TN, FN, custom_accuracy = custom_accuracy_metric(predictions_df)
# precision, recall, f1 = calculate_f1_score(TP, FP, TN, FN)
# print("Custom Accuracy:", custom_accuracy)
# print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")




# # Filter rows where conversation_ids equals 1
# filtered_data = predictions[predictions['conversation_ids'] == 16]
# print(filtered_data.shape)
# # # Save the filtered data to a CSV file
# filtered_data.to_csv('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/prediction_ux/test/prediction_conv_16.csv', index=False)


# TODO - work on the custom accuracy metric

