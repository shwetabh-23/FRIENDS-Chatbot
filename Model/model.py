from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm, trange
import time
from .utils import check_and_load_chkpts, save_chkpts
import torch    
import os

def train(train_dataloader, eval_loader, epochs, 
          lr, eps, save_dir, save_models_path, 
          save_tokens_path, config_name, tokenizer_name, model_name, device):

    t_total = len(train_dataloader.dataset)//epochs
    
    train_iterator = trange(0, epochs, desc = 'Epoch')
    tr_loss = 0
    global_step = 0
    model, tokenizer, optimizer, scheduler = check_and_load_chkpts(save_dir=save_dir, save_models_path=save_models_path, 
                                                                   save_tokens_path=save_tokens_path, config_name=config_name, 
                                                                     tokenizer_name=tokenizer_name, model_name=model_name, lr=lr, 
                                                                     eps=eps, t_total=t_total)
    
    model.zero_grad()

    train_loss = []
    test_loss = []
    for chkpt in train_iterator:
        start_time_1 = time.time()
        epoch_iterator = tqdm(train_dataloader, desc = 'Iteration')
        last_chkpt = None
        for step, batch in enumerate(epoch_iterator):
            if last_chkpt is not None:
                if chkpt < last_chkpt:
                    global_step += 1
                    continue
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.to(device)
            model.train()
            outputs = model(inputs, labels = labels)
            loss = outputs[0]
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            start_time_4 = time.time()
        print('loss for epoch {} : {}'.format(chkpt, tr_loss/global_step))
        perplexity, eval_loss = evaluate(model=model, eval_loader=eval_loader)

        print('Evaluation Loss for epoch {}: {}'.format(chkpt, eval_loss/global_step))

        train_loss.append(tr_loss/global_step)
        test_loss.append(eval_loss/global_step)
        
    save_chkpts(model=model, save_models_path=save_tokens_path, tokenizer=tokenizer, 
                save_tokens_path=save_tokens_path, save_dir=save_dir, 
                optimizer=optimizer, scheduler=scheduler)
    return train_loss, test_loss
    
def evaluate(eval_loader, model):
    
    eval_loss = 0
    model.eval()
    eval_steps = 0
    for batch in tqdm(eval_loader, desc = 'Evaluating'):
        #print(batch)
        inputs, labels = (batch, batch)
        with torch.no_grad():
            outputs= model(inputs, labels = labels)
            loss = outputs[0]
            eval_loss = loss.mean().item()
        eval_steps += 1
    eval_loss = eval_loss/ eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result = {'perplexity' : perplexity}
    return result, eval_loss