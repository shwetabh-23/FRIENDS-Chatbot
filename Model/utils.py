import os
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup

def check_and_load_chkpts(save_dir, save_models_path, save_tokens_path, 
                        config_name, tokenizer_name,
                        model_name, lr, eps, t_total):
    
    if (os.path.exists(save_models_path) and os.path.exists(save_tokens_path) and len(os.listdir(save_models_path)) > 0):
        print('Loading Pretrained Chkpts')
        all_saves = os.listdir(save_models_path)
        if len(all_saves) > 0:
            latest_save = all_saves[-1]
            last_chkpt = int(latest_save.split('-')[-1])
            print(' loading last chkpt', last_chkpt)
            config = AutoConfig.from_pretrained(config_name)

            model = AutoModelWithLMHead.from_pretrained(os.path.join(save_models_path, '{}-{}-{}'.format('model', 'checkpoint', last_chkpt)), config= config)
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_tokens_path, '{}-{}-{}'.format('token', 'checkpoint', last_chkpt)))
                    
            if os.path.isfile(os.path.join(save_dir, 'optimizer.pt')) and os.path.isfile(os.path.join(save_dir, 'scheduler.pt')):
                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameter = [
                {
                    'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                    'weight_decay' : 0.0
                },
                {
                    'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                    'weight_decay' : 0.0
                }]
                optimizer = AdamW(params= optimizer_grouped_parameter, lr= lr, eps=eps)
                scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=t_total)

                optimizer.load_state_dict(torch.load(os.path.join(save_dir, 'optimizer.pt')))
                scheduler.load_state_dict(torch.load(os.path.join(save_dir, 'scheduler.pt')))

                return model, tokenizer, optimizer, scheduler
    else:
        print('Starting training from scratch')
        config = AutoConfig.from_pretrained(config_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelWithLMHead.from_pretrained(model_name, config = config)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameter = [
        {
            'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay' : 0.0
        },
        {
            'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay' : 0.0
        }]
        optimizer = AdamW(params= optimizer_grouped_parameter, lr= lr, eps=eps)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=t_total)

        return model, tokenizer, optimizer, scheduler

def save_chkpts(model, save_models_path, tokenizer, save_tokens_path, save_dir, optimizer, scheduler):

    checkpoint_prefix = 'checkpoint'
    model_prefix = 'model'
    tokenizer_prefix = 'tokenizer'
    model.save_pretrained(os.path.join(save_models_path, '{}-{}-{}'.format(model_prefix, checkpoint_prefix, _)))
    tokenizer.save_pretrained(os.path.join(save_tokens_path, '{}-{}-{}'.format('token', 'checkpoint', _)))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(save_dir, 'scheduler.pt'))