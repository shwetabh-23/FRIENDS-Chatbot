import json
import pandas as pd
from Data import create_dataLoader
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig

with open('config.json', 'r') as f:
    hyperparameters = json.load(f)
    
data_path = hyperparameters['data_path']
n = hyperparameters['n']
batch_size = hyperparameters['batch_size']
model_name = hyperparameters['model_name']
config_name = hyperparameters['config_name']
tokenizer_name = hyperparameters['tokenizer_name']
SMOKE_TEST = hyperparameters['SMOKE_TEST']

data = pd.read_csv(data_path)

config = AutoConfig.from_pretrained(config_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelWithLMHead.from_pretrained(model_name, config = config)

train_dataloader, test_dataloader = create_dataLoader(data=data, batch_size=batch_size, n= n, tokenizer=tokenizer, SMOKE_TEST=SMOKE_TEST)
