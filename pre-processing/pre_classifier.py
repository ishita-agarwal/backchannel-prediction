import pickle
import numpy as np
import os

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
    num_words_spoken_so_far = np.array(pickle.load(open(f'{directory}num_words_spoken_so_far_conv_{conversation_id}.pkl', 'rb')))
    # num_words_spoken_so_far = [[sublist[0]] for sublist in num_words_spoken_so_far]
    # num_words_since_last_bc = np.array(pickle.load(open(f'{directory}num_words_since_last_bc_conv_{conversation_id}.pkl', 'rb')))
    # num_words_since_last_bc = [[sublist[0]] for sublist in num_words_since_last_bc]
    num_bc_so_far = np.array(pickle.load(open(f'{directory}num_bc_so_far_conv_{conversation_id}.pkl', 'rb')))
    # num_bc_so_far = [[sublist[0]] for sublist in num_bc_so_far]
    backchannel_rate_turn = np.array(pickle.load(open(f'{directory}backchannel_rate_turn_conv_{conversation_id}.pkl', 'rb')))
    # backchannel_rate_turn = [[sublist[0]] for sublist in backchannel_rate_turn]
    backchannel_rate_overall = np.array(pickle.load(open(f'{directory}backchannel_rate_overall_conv_{conversation_id}.pkl', 'rb')))
    # backchannel_rate_overall = [[sublist[0]] for sublist in backchannel_rate_overall]
    # print("num_words_spoken_so_far shape", num_words_spoken_so_far.shape)
    # if is_pickle_empty(f'{directory}/roberta_embedding_utterance.pkl'):
    #   continue
    embedding_utterance = pickle.load(open(f'/Users/ishitaagarwal/Documents/Embeddings/final/src/data/pca/roberta_embedding_conv_{conversation_id}.pkl', 'rb'))
    label = np.array(pickle.load(open(f'{directory}/label_conv_{conversation_id}.pkl', 'rb'))).reshape(-1, 1)
    
    # print("Label shape", label.shape)
    input_feature = np.array([num_words_spoken_so_far, num_bc_so_far, backchannel_rate_turn, backchannel_rate_overall]).T
    input_feature = np.squeeze(input_feature)
    # print("Input feature shape", input_feature.shape)
    # print("Embedding shape", embedding_utterance.shape)
    input_feature = np.hstack((input_feature, embedding_utterance))
    input_feature = np.hstack((input_feature, label))
    if (len(feature_matrix) == 0):
      feature_matrix = input_feature
    else:
      feature_matrix = np.vstack((feature_matrix, input_feature))


# label_matrix = label
with open('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/classifier_input/feature_matrix.pkl', 'wb') as f:
    pickle.dump(feature_matrix, f)

# with open('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/classifier_input/labels.pkl', 'wb') as f:
#     pickle.dump(label_matrix, f)