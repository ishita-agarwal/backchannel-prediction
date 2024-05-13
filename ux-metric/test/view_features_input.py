import pandas as pd
import numpy as np
import pickle

# Directory where the pickle files are stored
directory = f'/Users/ishitaagarwal/Documents/Embeddings/final/src/data/features'
conversation_id = '1'

# List of feature names stored in pickle files
features = [
    'utterance_parts', 'utterance_parts_wo_bc', 'num_words_spoken_so_far', 
    'num_words_since_last_bc', 'num_bc_so_far', 'backchannel_rate_turn', 
    'backchannel_rate_overall', 'conversation_ids', 'turn_ids', 'slice_ids',
    'label'
]

# Function to load data from a pickle file and assign column names
def load_pickle_data(feature_name):
    with open(f'{directory}/{feature_name}_conv_{conversation_id}.pkl', 'rb') as file:
        data = pickle.load(file)
    # Create a DataFrame and set column names based on the feature and column index
    column_names = [f"{feature_name}_{i}" for i in range(data.shape[1])] if len(data.shape) > 1 else [feature_name]
    return pd.DataFrame(data, columns=column_names)

# Load all features and concatenate them
all_features = [load_pickle_data(feature) for feature in features]
final_df = pd.concat(all_features, axis=1)  # Concatenate column-wise

# Save to CSV
final_csv_path = f'{directory}/test/all_features_conv_{conversation_id}.csv'
final_df.to_csv(final_csv_path, index=False)

print(f"All features saved to {final_csv_path}")
