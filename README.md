# Introduction
We present a machine learning model designed to predict the occurrence of backchanneling and interruptions in real-time conversations. We propose a new, more relevant user-experience metric that could better evaluate the models predicting the beginning of a backchannel.

# Dataset
The dataset we’ll be using for this project is the CANDOR corpus, which is a repository of conversations between humans in the form of audio, video and transcripts.

# Method
For classification, we use sklearn LogisticRegression binary classifier with L2 penalty and a tolerance of 1e-02. We also use an adjusted classification threshold of 0.07.

For splitting datasets between test and train in
classification, we split entire conversations (all the slices of turns belonging to a particular conversation ID) in either test or train to prevent overfitting.

We use 5-fold cross validation to evaluate our
results. We also propose a custom user-experience accuracy metric that allows 

# Results
We obtain an average AUC of 0.789.

# Running the code
Assuming you have the CANDOR dataset downloaded and extracted all conversation raw data in csv format for Audiophile and Backbiter.

Steps:
1. get_features.py (to generate all features, excluding embeddings)
2. pre_classifier.py (to generate feature , label matrix for classifier input)
3. default-metric/logistic_regression.py, default-metric/logistic_regression_kfold.py (classifier with default evaluation)
4. ux-metric/logistic_regression_ux.py, default-metric/logistic_regression_ux_kfold.py (classifier with custom user experience evaluation)

The above steps assumes embeddings for conversations are already generated and PCA has been applied on them to create a smaller (50) dimension embedding.

# Generating embeddings
Script to get RoBERTa embeddings: get_embeddings.py - it uses CUDA and can be run on GPU.

Embeddings are created before pre_classifier.py is run.
To run PCA on embeddings,
1. pre_pca.py
2. pca_embeddings.py