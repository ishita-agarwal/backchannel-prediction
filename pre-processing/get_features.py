import pandas as pd
import numpy as np

import os

import sys
import pickle

entire_backbiter_dataset = pd.read_csv('/data/backbiter.csv')
entire_audiophile_dataset = pd.read_csv('/data/audiophile.csv')

start_conv_index = 1
max_conv_id = 1536
pca_input_embedding = []  # this will finally be a np.array with each row representing an embedding along with an extra column containing conversation ID

for conversation_id in range(start_conv_index, max_conv_id+1):
    if conversation_id in [159, 1230]:
      continue
    print("Generating features for conversation ID ", conversation_id)
    directory = f'/data/features'
    backbiter_conversation_df = entire_backbiter_dataset[entire_backbiter_dataset['conversation_id'] == conversation_id].sort_values(by='turn_id')
    audiophile_conversation_df = entire_audiophile_dataset[entire_audiophile_dataset['conversation_id'] == conversation_id].sort_values(by='turn_id')

    speaker = backbiter_conversation_df['speaker'].iloc[0]  # First speaker - speaker, second - backchanneler
    backchanneler = backbiter_conversation_df['speaker'].iloc[1]  # todo - assumption - alternate dialogs spoken by different speakers. Only 2 speakers in conversation.

    utterances = []
    non_bc_utterances = []
    
    # iterating through each line (turn) in backbiter conversation
    
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
    conversation_ids = []  # will store conversation id for each corresponding slice
    turn_ids = []
    turn_id = 0
    slice_ids = []

    # todo - the number of rows in utterance without backchannel will be different from utterance with backchannel - how to deal with that?
    for utterance in utterances:
        total_words_spoken = 0
        words_since_last_bc = 0
        bc_count = 0
        sentence = ""
        sentence_no_bc = ""
        bc_right_now = False
        slice_id = 0
        for word in utterance:  # iterating through each slice in a turn
            curr_label = False
            total_words_spoken += 1
            total_words_spoken_overall += 1
            num_words_spoken_so_far.append([total_words_spoken])
            sentence += word + " "
            num_words_since_last_bc.append([words_since_last_bc])
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
            
            utterance_parts.append(sentence[:-1])
            utterance_parts_wo_bc.append(sentence_no_bc[:-1])
            # num_bc_so_far = np.concatenate((num_bc_so_far, np.array(bc_count)))
            num_bc_so_far.append([bc_count])
            backchannel_rate_overall.append([total_bc_spoken_overall/total_words_spoken_overall])
            backchannel_rate.append([bc_count/total_words_spoken])
            label.append(curr_label)
            turn_ids.append(turn_id)
            slice_ids.append(slice_id)
            slice_id += 1
            conversation_ids.append(conversation_id)
        turn_id += 1

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{directory}/utterance_parts_conv_{conversation_id}.pkl', 'wb') as f:
       pickle.dump(np.array(utterance_parts), f)

    with open(f'{directory}/utterance_parts_wo_bc_conv_{conversation_id}.pkl', 'wb') as f:
       pickle.dump(np.array(utterance_parts_wo_bc), f)

    with open(f'{directory}/num_words_spoken_so_far_conv_{conversation_id}.pkl', 'wb') as f:
       num_words_spoken_so_far = np.array(num_words_spoken_so_far)
       pickle.dump(num_words_spoken_so_far, f)

    # with open(f'{directory}/num_words_spoken_overall_conv_{conversation_id}.pkl', 'wb') as f:
    #    total_words_spoken_overall = np.array(total_words_spoken_overall)
    #    pickle.dump(total_words_spoken_overall, f)

    with open(f'{directory}/num_words_since_last_bc_conv_{conversation_id}.pkl', 'wb') as f:
       num_words_since_last_bc = np.array(num_words_since_last_bc)
       pickle.dump(num_words_since_last_bc, f)

    with open(f'{directory}/num_bc_so_far_conv_{conversation_id}.pkl', 'wb') as f:
       num_bc_so_far = np.array(num_bc_so_far)
    #    print("num_bc_so_far shape", num_bc_so_far.shape)
       pickle.dump(num_bc_so_far, f)

    with open(f'{directory}/backchannel_rate_turn_conv_{conversation_id}.pkl', 'wb') as f:
       backchannel_rate = np.array(backchannel_rate)
    #    print("backchannel_rate shape", backchannel_rate.shape)
       pickle.dump(backchannel_rate, f)

    with open(f'{directory}/backchannel_rate_overall_conv_{conversation_id}.pkl', 'wb') as f:
       backchannel_rate_overall = np.array(backchannel_rate_overall)
       pickle.dump(backchannel_rate_overall, f)
    
    with open(f'{directory}/conversation_ids_conv_{conversation_id}.pkl', 'wb') as f:
        conversation_ids = np.array(conversation_ids)
        pickle.dump(conversation_ids, f)

    with open(f'{directory}/turn_ids_conv_{conversation_id}.pkl', 'wb') as f:
       turn_ids = np.array(turn_ids)
       pickle.dump(turn_ids, f)

    with open(f'{directory}/slice_ids_conv_{conversation_id}.pkl', 'wb') as f:
       slice_ids = np.array(slice_ids)
       pickle.dump(slice_ids, f)

    with open(f'{directory}/label_conv_{conversation_id}.pkl', 'wb') as f:
       label = np.array(label)
       pickle.dump(label, f)

