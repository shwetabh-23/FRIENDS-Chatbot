import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

def remove_inside_brackets(input_string):
    pattern = r'\([^)]*\)'
    result_string = re.sub(pattern, '', input_string)

    return result_string

def preprocessing(data):
    data = data.dropna()
    temp = data['Speaker'].unique()
    data['Speaker'] = data['Speaker'].apply(lambda x : x.lower())
    speakers = ['monica', 'joey', 'ross', 'rachel', 'chandler', 'phoebe']
    data = data[data['Speaker'].isin(speakers)]
    data.reset_index(inplace = True)
    return data

def create_context(data, n):

    all_episode = data['Episode'].unique()
    contexted = []
    for episode in all_episode:
        #print(episode)
        temp_data = data[data['Episode'] == episode]
        temp_data.reset_index(inplace=True)
        #print(temp_data)
        for i in range(n, len(temp_data)):
            row = []
            prev = i-1-n
            for j in range(i, prev, -1):
                #print(temp_data.iloc[j]['Text'])
                row.append(temp_data.iloc[j]["Speaker"] + ' : ' + temp_data.iloc[j]['Text'])
            contexted.append(row)
            
    columns = ['response', 'context'] 
    columns = columns + ['context/'+str(i) for i in range(n-1)]

    new_data = pd.DataFrame(contexted, columns=columns)

    return new_data

def construct_conv(row, tokenizer, eos = True):
    #print(row)
    flatten = lambda l: [item for sublist in l for item in sublist]
    #print([x for x in row])
    #print([tokenizer.encode(x) for x in row])
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row[1:]]))
    #print(conv)
    conv = flatten(conv)
    return conv

class dataset(Dataset):
    def __init__(self, tokenizer, data, block_size = 512):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.examples = []
        for _, row in data.iterrows():
            conv = construct_conv(row, tokenizer, eos = True)
            self.examples.append(conv)
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, indx):
        return torch.tensor(self.examples[indx], dtype = torch.long)

def create_dataLoader(data, batch_size, n, tokenizer, SMOKE_TEST):
    if SMOKE_TEST:
        data = data.iloc[0:100]
    data['Text'] = data['Text'].apply(lambda x : remove_inside_brackets(x))
    data = preprocessing(data)
    data = create_context(data, n)

    def collate(examples):
        return pad_sequence(examples, batch_first = True, padding_value = tokenizer.eos_token_id)
    
    trn_df, val_df = train_test_split(data, test_size=0.2)
    train_Set = dataset(tokenizer=tokenizer, data=trn_df)
    val_set = dataset(tokenizer=tokenizer, data= val_df)
    
    train_dataloader = DataLoader(train_Set,
                                  batch_size = batch_size, collate_fn= collate, drop_last= True)
    test_dataloader = DataLoader(val_set,
                                  batch_size = batch_size, collate_fn= collate, drop_last= True)
    return train_dataloader, test_dataloader