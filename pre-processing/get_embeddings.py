import pandas as pd
import numpy as np
import os
import pickle
from transformers import RobertaTokenizer, RobertaModel
import torch


class RoBERTa:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')

        if torch.cuda.is_available():
            self.model.to("cuda")

    def encode(self, texts, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                if torch.cuda.is_available():
                    encoded_input = {k: v.to("cuda") for k, v in encoded_input.items()}

                outputs = self.model(**encoded_input)
                sentence_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move the data back to CPU if necessary
                all_embeddings.append(sentence_embeddings)

        return np.concatenate(all_embeddings, axis=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model is using device: {device}")

entire_backbiter_dataset = pd.read_csv('/src/data/raw/backbiter.csv')
entire_audiophile_dataset = pd.read_csv('/src/data/raw/audiophile.csv')

roberta_model = RoBERTa()
# n_pca_components = 50
# pca = PCA(n_components=n_pca_components)  # todo - increase to 50 when num_samples > 50
start_conv_index = 1231
max_conv_id = 1536
pca_input_embedding = []  # this will finally be a np.array with each row representing an embedding along with an extra column containing conversation ID
conversation_ids = []  # will store conversation id for each corresponding embedding

for conversation_id in range(start_conv_index, max_conv_id+1):
    #print("Generating embedding for conversation ID ", conversation_id)
    directory = f'data/input'
    backbiter_conversation_df = entire_backbiter_dataset[entire_backbiter_dataset['conversation_id'] == conversation_id].sort_values(by='turn_id')
    audiophile_conversation_df = entire_audiophile_dataset[entire_audiophile_dataset['conversation_id'] == conversation_id].sort_values(by='turn_id')

    speaker = backbiter_conversation_df['speaker'].iloc[0]  # First speaker - speaker, second - backchanneler
    backchanneler = backbiter_conversation_df['speaker'].iloc[1]  # todo - assumption - alternate dialogs spoken by different speakers. Only 2 speakers in conversation.

    utterances = []
    non_bc_utterances = []
    # iterating through each line in backbiter conversation
    for index, line_backbiter in backbiter_conversation_df.iterrows():
        if line_backbiter['speaker'] == speaker:
            non_bc_utterances.append(line_backbiter['utterance'])
            if pd.isna(backbiter_conversation_df.at[index, 'interval']):
                utterances.append(line_backbiter['utterance'])
            else:
                # there exists some backchannel
                start_time = line_backbiter['start']
                end_time = line_backbiter['stop']
                filter = (audiophile_conversation_df['start'] >= start_time) & (audiophile_conversation_df['stop'] <= end_time)
                filtered_rows = audiophile_conversation_df[filter].sort_values(by='turn_id')
                utterance = ""
                for i, line_audiophile in filtered_rows.iterrows():
                    if line_audiophile['speaker'] == speaker:
                        utterance += line_audiophile['utterance']
                    elif line_audiophile['speaker'] == backchanneler:
                        # it is a backchannel
                        utterance += " <" + line_audiophile['utterance'] + "> "
                utterances.append(utterance)
    utterance_initial = utterances.copy()
    utterances = []

    non_bc_utterance_initial = non_bc_utterances.copy()
    non_bc_utterances = []

    for utterance in utterance_initial:
    # split it by space
        utterances.append(utterance.split())

    for utterance in non_bc_utterance_initial:
        non_bc_utterances.append(utterance.split())

    # now we have utterances, where each row is a list of words spoken by speaker in a backbiter turn
    utterance_parts = []
    utterance_parts_wo_bc = []
    embedding_utterance = [] #embedding of non-backchannel turn words (i.e backbiter turns)
    num_words_spoken_so_far = []
    num_words_since_last_bc = []
    num_bc_so_far = []   # in the turn
    backchannel_rate = []   # in the turn
    backchannel_rate_overall = []   # overall, across the entire conversation
    total_words_spoken_overall = 0   # overall, across the entire conversation
    total_bc_spoken_overall = 0     # overall, across the entire conversation
    label = []  # is last word BC

    for utterance in utterances:
        total_words_spoken = 0
        words_since_last_bc = 0
        bc_count = 0
        sentence = ""
        sentence_no_bc = ""
        bc_right_now = False

        for word in utterance:
            curr_label = False
            total_words_spoken += 1
            total_words_spoken_overall += 1
            num_words_spoken_so_far.append([total_words_spoken])
            sentence += word + " "

            if (word[0] == '<'):            # if current word is backchannel start
                bc_count += 1
                total_bc_spoken_overall += 1
                words_since_last_bc = 0
                curr_label = True
                bc_right_now = True
            else:
                words_since_last_bc += 1

            if (bc_right_now == False):
                sentence_no_bc += word + " "
            if (word[-1] == '>'):
                bc_right_now = False
            num_words_since_last_bc.append([words_since_last_bc])
            utterance_parts.append(sentence[:-1])
            utterance_parts_wo_bc.append(sentence_no_bc[:-1])
            # num_bc_so_far = np.concatenate((num_bc_so_far, np.array(bc_count)))
            num_bc_so_far.append([bc_count])
            backchannel_rate_overall.append([total_bc_spoken_overall/total_words_spoken_overall])
            backchannel_rate.append([bc_count/total_words_spoken])
            label.append(curr_label)


    # roberta model
    embedding_utterance = roberta_model.encode(utterance_parts_wo_bc)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{directory}/roberta_embedding_conv_{conversation_id}.pkl', 'wb') as f:
        pickle.dump(embedding_utterance, f)

print("done generating embeddings")