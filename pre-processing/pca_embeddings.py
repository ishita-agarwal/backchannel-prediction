import pickle
from sklearn.decomposition import PCA

n_pca_components = 50
pca = PCA(n_components=n_pca_components)
start_conv_index = 1
MAX_CONV_ID = 1536

with open('/final/src/data/pca_input/pca_fit_embeddings.pkl', 'rb') as f:
    pca_input_embedding = pickle.load(f)


print("Embedding shape: ", pca_input_embedding.shape)

pca.fit(pca_input_embedding)

# filter on conversation ID
for conversation_id in range(start_conv_index, MAX_CONV_ID+1):
  if conversation_id in [159, 1230]:
    continue
  embedding_utterance = pickle.load(open(f'/final/src/data/input/roberta_embedding_conv_{conversation_id}.pkl', 'rb'))
  embedding_utterance_pca = pca.transform(embedding_utterance)
  with open(f'/final/src/data/pca/roberta_embedding_conv_{conversation_id}.pkl', 'wb') as f:
    pickle.dump(embedding_utterance_pca, f)
  