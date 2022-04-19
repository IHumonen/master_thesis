import os

import torch
import wandb

import pandas as pd

def train(model, 
            data,
            indices,
            idx2token, 
            optimizer, 
            criterion,
            epoch,
            device):

    print(f"epoch {epoch}")

    epoch_loss = 0
    model.train()
    
    optimizer.zero_grad()

    predictions = torch.index_select(model(data.to(device)).to(device), 0, indices)
    ys = torch.index_select(torch.tensor(list(idx2token.keys())).to(device), 0, indices)
    
    print(predictions.shape, ys.shape)

    loss = criterion(predictions.view(-1, predictions.size(-1)), ys.view(-1))
    loss.backward()

    optimizer.step()

    batch_loss = loss.item()
    epoch_loss = batch_loss
    
    return epoch_loss

def evaluate(model, 
                data,
                indices,
                idx2token, 
                criterion, 
                epoch,
                device,
                timestamp,
                save_checkpoints=False
                ):
    epoch_loss = 0

    #    model.train(False)
    model.eval()

    with torch.no_grad():

        predictions = torch.index_select(model(data.to(device)).to(device), 0, indices)
        ys = torch.index_select(torch.tensor(list(idx2token.keys())).to(device), 0, indices)

        loss = criterion(predictions.view(-1, predictions.size(-1)), ys.view(-1))

        epoch_loss += loss.item()

    if save_checkpoints:
        file_name = 'epoch_' + str(epoch) + '_' + timestamp + '.pt'
        path = os.path.expanduser("".join(['logs/', file_name]))
        torch.save(model.state_dict(), path)

        print(f"epoch loss {epoch_loss}")
    print(
        "========================================================================================================")

    
    return epoch_loss


def get_item(d, key):
    if key in d:
        return d[key]
    else:
        return None 

def update_log():
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("kesha_humonen/master_thesis")

    summary_list, config_list, name_list, created_list = [], [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
              if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        
        created_list.append(run._attrs['createdAt'])

    runs_df = pd.DataFrame({
        "summary" : summary_list,
        "config" : config_list,
        "name" : name_list,
        'created' : created_list
        })


    for key in summary_list[0]:
        runs_df[key] = runs_df['summary'].apply(get_item, args=(key,))
    for key in config_list[0]:
        runs_df[key] = runs_df['config'].apply(get_item, args=(key,))


    runs_df.to_csv("wandb_log.csv")
    
    return runs_df
