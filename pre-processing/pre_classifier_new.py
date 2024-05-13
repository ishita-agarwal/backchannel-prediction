import pickle
import numpy as np
import os
import pandas as pd

def is_pickle_empty(file_path):
    return os.path.getsize(file_path) == 0

# Load dataset
max_conv_id = 1536
feature_matrix = []
label_matrix = np.empty(0)

label = np.empty(0)
index = 0
for conversation_id in range(1, max_conv_id+1):
    # print(conversation_id)
    if conversation_id in [159, 1230]:
      continue
    directory = f'/Users/ishitaagarwal/Documents/Embeddings/final/src/data/features/'
    num_words_spoken_so_far = np.array(pickle.load(open(f'{directory}num_words_spoken_so_far_conv_{conversation_id}.pkl', 'rb'))).flatten()
    num_words_since_last_bc = np.array(pickle.load(open(f'{directory}num_words_since_last_bc_conv_{conversation_id}.pkl', 'rb'))).flatten()
    num_bc_so_far = np.array(pickle.load(open(f'{directory}num_bc_so_far_conv_{conversation_id}.pkl', 'rb'))).flatten()
    backchannel_rate_turn = np.array(pickle.load(open(f'{directory}backchannel_rate_turn_conv_{conversation_id}.pkl', 'rb'))).flatten()
    backchannel_rate_overall = np.array(pickle.load(open(f'{directory}backchannel_rate_overall_conv_{conversation_id}.pkl', 'rb'))).flatten()
    slice_ids = np.array(pickle.load(open(f'{directory}slice_ids_conv_{conversation_id}.pkl', 'rb')))
    turn_ids = np.array(pickle.load(open(f'{directory}turn_ids_conv_{conversation_id}.pkl', 'rb')))
    conversation_ids = np.array(pickle.load(open(f'{directory}conversation_ids_conv_{conversation_id}.pkl', 'rb')))
    embedding_utterance = pickle.load(open(f'/Users/ishitaagarwal/Documents/Embeddings/final/src/data/pca/roberta_embedding_conv_{conversation_id}.pkl', 'rb'))
    label = np.array(pickle.load(open(f'{directory}/label_conv_{conversation_id}.pkl', 'rb'))).reshape(-1, 1)

    # print("Label shape", label.shape)
    input_feature = np.array([num_words_spoken_so_far, num_words_since_last_bc, num_bc_so_far, backchannel_rate_turn, backchannel_rate_overall, slice_ids, turn_ids, conversation_ids]).T
    input_feature = np.squeeze(input_feature)
    
    # print("Embedding shape", embedding_utterance.shape)
    input_feature = np.hstack((input_feature, embedding_utterance))
    input_feature = np.hstack((input_feature, label))
    print("Input feature shape", input_feature.shape)
    if (len(feature_matrix) == 0):
      feature_matrix = input_feature
    else:
      feature_matrix = np.vstack((feature_matrix, input_feature))

embedding_dim = 50  # Example embedding dimension

# Define the column names
feature_names = ['num_words_spoken_so_far', 'num_words_since_last_bc', 'num_bc_so_far', 
                 'backchannel_rate_turn', 'backchannel_rate_overall', 
                 'slice_ids', 'turn_ids', 'conversation_ids'] \
              + [f'embedding_{i+1}' for i in range(embedding_dim)] \
              + ['label']

# Convert feature_matrix to DataFrame
df = pd.DataFrame(feature_matrix, columns=feature_names)

# Print the shape of the DataFrame to confirm conversion
print("DataFrame shape:", df.shape)

# Save DataFrame to a pickle file
df.to_pickle('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/classifier_input_ux/feature_matrix.pkl')
