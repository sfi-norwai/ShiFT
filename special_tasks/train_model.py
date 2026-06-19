
import random
import numpy as np
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset
from tqdm import tqdm

class SupervisedPretrainModel(nn.Module):
    def __init__(self, feature_extractor, feature_dim, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor.net
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, feature_dim]
        features = F.max_pool1d(
                features.transpose(1, 2),
                kernel_size = features.size(1),
            ).transpose(1, 2).squeeze()

        logits = self.classifier(features)    # [B, num_classes]
        return logits

class SupervisedPretrainModelPool(nn.Module):
    def __init__(self, feature_extractor, feature_dim, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor.net
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, feature_dim]
        logits = self.classifier(features)    # [B, num_classes]
        return logits

def create_subset_dataloader(dataset, fraction, batch_size, config, shuffle=True):
    """
    Create a DataLoader for a random subset of the dataset.
    
    Args:
        dataset: The full dataset object.
        fraction: Fraction of the dataset to select (e.g., 0.01 for 1%).
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the dataset when creating the subset.

    Returns:
        DataLoader for the subset of the dataset.
    """
    # Determine subset size
    subset_size = int(len(dataset) * fraction)
    
    # Generate indices for the subset
    indices = torch.randperm(len(dataset))[:subset_size]
    
    # Create the subset
    subset = Subset(dataset, indices)
    
    # Create a DataLoader for the subset
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=config.NUM_WORKERS,)
    return dataloader

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return correct / total if total > 0 else 0.0

def augment_batch(x, jitter_std=0.01, scale_std=0.1, time_mask_p=0.2, ch_dropout_p=0.1):
    # x: numpy or torch (B, T, C)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    # jitter
    x = x + torch.randn_like(x) * jitter_std
    # scaling
    B, T, C = x.shape
    scales = 1.0 + torch.randn(B, 1, 1) * scale_std
    x = x * scales
    # channel dropout
    if ch_dropout_p > 0:
        mask = torch.rand(B, 1, C) > ch_dropout_p
        x = x * mask.to(x.dtype)
    # time masking (random contiguous block)
    if time_mask_p > 0:
        for i in range(B):
            if random.random() < time_mask_p:
                t0 = random.randint(0, T//2)
                tlen = random.randint(1, T//4)
                x[i, t0:t0+tlen, :] = 0.0
    return x

def train_supervised_pretrain(model, train_dataset, num_labels, args, config, device):

    wrapped_model = SupervisedPretrainModel(
                        feature_extractor=model,
                        feature_dim=args['out_features'],
                        num_classes=num_labels
                    )
    epochs = args['supervised_epochs']
    wrapped_model = wrapped_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=float(args['lr']), weight_decay=float(args['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.config.PATIENCE)


    # train_loader = create_subset_dataloader(train_dataset, args['data_perc'], args['batch_size'], config)

    train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args['batch_size'],
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = False
            )
    
    for epoch in tqdm(range(epochs)):
        wrapped_model.train()
        running_loss = 0.0
        for x, y in train_loader:
            
            x, y = x.to(device).float(), y.to(device).long()
            
            optimizer.zero_grad()
            
            logits = wrapped_model(x)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()

    return wrapped_model

def train_supervised(model, train_dataset, args, config, device):

    epochs = args['supervised_epochs']
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args['lr']), weight_decay=float(args['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.config.PATIENCE)


    # train_loader = create_subset_dataloader(train_dataset, args['data_perc'], args['batch_size'], config)

    train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args['batch_size'],
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = False
            )
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            
            x, y = x.to(device).float(), y.to(device).long()
            
            optimizer.zero_grad()
            
            logits = model(x)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()

    return model

def train_loop(model, train_dataset, valid_dataset, args, config, device, warmup_pct=0.10, early_stop_patience=10):

    train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args['batch_size'],
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = False
            )
    
    val_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=args['batch_size'],
                shuffle = False,
                num_workers=config.NUM_WORKERS,
                drop_last = False,
            )
    
    model.to(device)
    epochs = args['epochs']
    optimizer = AdamW(model.parameters(), lr=float(args['lr']), weight_decay=float(args['weight_decay']))
    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_pct * total_steps)

    

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    patience = 0
    step = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:

            # augmentation
            x = augment_batch(x)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}  TrainLoss: {avg_loss:.4f}  ValAcc: {val_acc:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        # early stopping
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_model = model
            patience = 0
            # torch.save(model.state_dict(), "best_model.pt")
        else:
            patience += 1
            if patience >= early_stop_patience:
                print("Early stopping triggered.")
                break

    print("Best val acc:", best_acc)
    return best_model

def train_supervised_pretrain_pool(model, train_dataset, num_labels, args, config, device):

    wrapped_model = SupervisedPretrainModelPool(
                        feature_extractor=model,
                        feature_dim=args['out_features'],
                        num_classes=num_labels
                    )
    epochs = args['supervised_epochs']
    wrapped_model = wrapped_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=float(args['lr']), weight_decay=float(args['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.config.PATIENCE)


    # train_loader = create_subset_dataloader(train_dataset, args['data_perc'], args['batch_size'], config)

    train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args['batch_size'],
                shuffle = True,
                # num_workers=config.NUM_WORKERS,
                drop_last = False
            )

    for epoch in tqdm(range(epochs)):
        wrapped_model.train()
        running_loss = 0.0
        for x, y in train_loader:

            x, y = x.to(device).float(), y.to(device).long()

            optimizer.zero_grad()

            logits = wrapped_model(x)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

    return wrapped_model

def train_supervised_pretrain_catt(model, train_dataset, num_labels, args, config, device):

    wrapped_model = SupervisedPretrainModel(
                        feature_extractor=model,
                        feature_dim=args['out_features'],
                        num_classes=num_labels
                    )
    epochs = args['supervised_epochs']
    wrapped_model = wrapped_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=float(args['lr']), weight_decay=float(args['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.config.PATIENCE)


    # train_loader = create_subset_dataloader(train_dataset, args['data_perc'], args['batch_size'], config)

    train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args['batch_size'],
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = False
            )
    
    for epoch in tqdm(range(epochs)):
        wrapped_model.train()
        running_loss = 0.0
        for x, y in train_loader:
            
            x, y = x.to(device).float(), y.to(device).long()

            x = x.view(-1, x.size(3), x.size(2))
            y = y.view(-1)
            
            optimizer.zero_grad()
            
            logits = wrapped_model(x)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()

    return wrapped_model
