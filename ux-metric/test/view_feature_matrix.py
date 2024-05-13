import pandas as pd
import pickle
import numpy as np

# Load the feature matrix from the pickle file
with open('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/classifier_input_ux/feature_matrix.pkl', 'rb') as f:
    feature_matrix = pickle.load(f)

# Assuming the last column is the label and the previous columns include feature data and embedding
# Let's say you know there are 768 embedding dimensions (you will need to adjust this number)
# num_embedding_dimensions = 50  # adjust this number based on your actual embedding size

# Exclude embedding dimensions from the feature matrix, keep only the original features and the label
# Adjust -769 to account for the number of embedding dimensions and one label column
# feature_matrix_without_embedding = feature_matrix[:, :-num_embedding_dimensions - 1]

# Define column names based on the features without embedding dimensions
column_names = ['num_words_spoken_so_far', 'num_words_since_last_bc', 'num_bc_so_far', 'backchannel_rate_turn', 'backchannel_rate_overall', 
                'slice_ids', 'turn_ids', 'conversation_ids', 'label']

# Create a DataFrame
df = pd.DataFrame(feature_matrix, columns=column_names)

# Save to CSV
csv_file_path = '/Users/ishitaagarwal/Documents/Embeddings/final/src/data/classifier_input_ux/test/feature_matrix_no_embeddings_conv_1.csv'
df.to_csv(csv_file_path, index=False)

print(f"Feature matrix without embeddings saved to CSV at {csv_file_path}")
