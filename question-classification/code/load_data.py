import os
import pickle
from transformers import BertTokenizer, BertModel
import torch 
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def question_to_embeddings(question):

    input_tokens = torch.tensor(tokenizer.encode(question))[:512].unsqueeze(0)
    outputs = model(input_tokens.long())

    last_hidden_states = outputs[0]
    
    final_input = last_hidden_states
    
    if 512-last_hidden_states.shape[1] != 0:
        pad = torch.zeros(1, 512-last_hidden_states.shape[1], 768)
        final_input = torch.cat((last_hidden_states, pad), dim=1)

    final_input = final_input.detach().numpy()
    return final_input

def store_questions(filename, data_type):
    labels = {'ABBR': 0, 'ENTY': 1, 'DESC': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}

    questions = {}

    counter = 0

    with open('../data/' + filename, 'r', encoding="ISO-8859-1") as question_file:
        for question in question_file:
            question = question.rstrip().split(' ')
            question_class = question[0].split(':')[0]
            questions[counter] = {}

            label_vector = np.zeros(6)
            label_vector[labels[question_class]] = 1

            questions[counter]['label'] = label_vector
            questions[counter]['question'] = question_to_embeddings(question[1:])
            if counter % 1 == 0:
                print(f'Question {counter}/5500')
                with open(f'../data/{data_type}_pre_processed/{data_type}_{counter}.pickle', 'wb') as pickle_file:
                    pickle.dump(questions, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                questions = {}
            counter += 1

def pre_process_data():

    store_questions('masked_questions.label', 'experiment')
    
pre_process_data()

