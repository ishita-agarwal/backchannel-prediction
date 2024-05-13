# -*- coding: utf-8 -*-
"""pca_embeddings.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11StdeyeOZs5swjxWFwGcxNxKClPOVuwf
"""

import pandas as pd
import numpy as np

import os
import sys
import pickle
import torch
from sklearn.decomposition import PCA

n_pca_components = 50
pca = PCA(n_components=n_pca_components)  # todo - increase to 50 when num_samples > 50
start_conv_index = 1
MAX_CONV_ID = 1536

with open('/Users/ishitaagarwal/Documents/Embeddings/final/src/data/pca_input/pca_fit_embeddings.pkl', 'rb') as f:
    pca_input_embedding = pickle.load(f)


print("Embedding shape: ", pca_input_embedding.shape)

pca.fit(pca_input_embedding)

# # todo- separate. fit pca on a random sample of input embeddings
# embedding_utterance_pca = pca.transform(pca_input_embedding)
# # do transform for any amount of input points, result wont change
# print("Embedding after PCA shape: ", embedding_utterance_pca.shape)

# embedding_df = pd.DataFrame(embedding_utterance_pca)
# embedding_df['conversation_id'] = conversation_ids


# filter on conversation ID
for conversation_id in range(start_conv_index, MAX_CONV_ID+1):
  if conversation_id in [159, 1230]:
    continue
  embedding_utterance = pickle.load(open(f'/Users/ishitaagarwal/Documents/Embeddings/final/src/data/input/roberta_embedding_conv_{conversation_id}.pkl', 'rb'))
  embedding_utterance_pca = pca.transform(embedding_utterance)
  with open(f'/Users/ishitaagarwal/Documents/Embeddings/final/src/data/pca/roberta_embedding_conv_{conversation_id}.pkl', 'wb') as f:
    pickle.dump(embedding_utterance_pca, f)
  
