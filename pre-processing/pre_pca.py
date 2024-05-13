"""Pre-processing"""
import numpy as np

import pickle
import random


start_conv_index = 1
MAX_CONV_ID = 1536
# random.seed(42)

# Select 1000 random conversations from all conversations

excluded_numbers = [159, 1230]
random_integers = []
while len(random_integers) < 1000:
    rand_int = random.randint(start_conv_index, MAX_CONV_ID)
    if rand_int not in excluded_numbers:
        random_integers.append(rand_int)

pca_input_embedding = []  # this will finally be a np.array with each row representing an embedding along with an extra column containing conversation ID
for conversation_id in random_integers:

  embedding_utterance = pickle.load(open(f'/final/src/data/input/roberta_embedding_conv_{conversation_id}.pkl', 'rb'))
  EMBEDDING_LEN = embedding_utterance.shape[0]

  # Select 50 random embeddings from each conversation
  embedding_indices = [random.randint(0, EMBEDDING_LEN-1) for _ in range(50)]

  embedding_subarray = embedding_utterance[embedding_indices]

  if (len(pca_input_embedding) == 0):
    pca_input_embedding = embedding_subarray
  else:
    pca_input_embedding = np.vstack((pca_input_embedding, embedding_subarray))

print(pca_input_embedding.shape)

with open('pca_fit_embeddings.pkl', 'wb') as f:
    pickle.dump(pca_input_embedding, f)

