import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def eval_supervised(model, valid_dataset, feature_dim, num_epochs, batch_size, config, eval_protocol='linear'):

    val_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                shuffle = False,
                drop_last = False
            )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapped_model = model.wrapped_model.to(device)

    for epoch in range(num_epochs):

        # --- Validation ---
        wrapped_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device).float(), y_val.to(device).long()
                logits = wrapped_model(x_val)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y_val.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        val_acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        return [val_acc*100, f1, precision, recall]
    

def eval_supervised2(model, valid_dataset, feature_dim, num_epochs, batch_size, config, eval_protocol='linear'):

    val_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                shuffle = False,
                drop_last = False
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapped_model = model.to(device)

    for epoch in range(num_epochs):

        # --- Validation ---
        wrapped_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device).float(), y_val.to(device).long()
                logits = wrapped_model(x_val)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y_val.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        val_acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        return [val_acc*100, f1, precision, recall]

def eval_supervised_catt(model, valid_dataset, feature_dim, num_epochs, batch_size, config, eval_protocol='linear'):

    val_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                shuffle = False,
                num_workers=config.NUM_WORKERS,
                drop_last = False,
            )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    for epoch in range(num_epochs):

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device).float(), y_val.to(device).long()

                x_val = x_val.view(-1, x_val.size(3), x_val.size(2))
                y_val = y_val.view(-1)

                logits = model(x_val)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y_val.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        val_acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        return [val_acc*100, f1, precision, recall]
