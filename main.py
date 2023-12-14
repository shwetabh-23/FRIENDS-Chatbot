import json
import pandas as pd
from Data import create_dataLoader
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
from Model import train
import torch
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

with open('config.json', 'r') as f:
    hyperparameters = json.load(f)
    
data_path = hyperparameters['data_path']
n = hyperparameters['n']
batch_size = hyperparameters['batch_size']
model_name = hyperparameters['model_name']
config_name = hyperparameters['config_name']
tokenizer_name = hyperparameters['tokenizer_name']
SMOKE_TEST = hyperparameters['SMOKE_TEST']
lr = float(hyperparameters['lr'])
eps = float(hyperparameters['eps'])
epochs = int(hyperparameters['epochs'])
save_dir = hyperparameters['save_dir']
save_models_dir = hyperparameters['saved_model_dir']
save_tokens_dir = hyperparameters['saved_tokens_dir']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
data = pd.read_csv(data_path)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
train_dataloader, test_dataloader = create_dataLoader(data=data, batch_size=batch_size, n= n, tokenizer=tokenizer, SMOKE_TEST=SMOKE_TEST)

train_loss, test_loss = train(train_dataloader=train_dataloader, eval_loader=test_dataloader, epochs=epochs, 
                              lr=lr, eps=eps, save_dir=save_dir, save_models_path=save_models_dir, 
                              save_tokens_path=save_tokens_dir, config_name=config_name, tokenizer_name=tokenizer_name,
                                model_name=model_name,device=device)