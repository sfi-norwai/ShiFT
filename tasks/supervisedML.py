import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MiniRocketMultivariateVariable,
)

def encode_raw_dataset(dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size)
    reps = []
    labs = []
    with torch.no_grad():
        for data, labels in loader:
            data = data
            rep = data.mean(dim=1)            # (B, C) fast pooling

            reps.append(rep)
            labs.append(labels)

    reps = torch.cat(reps, dim=0).numpy()
    labs = torch.cat(labs, dim=0).numpy()

    return reps, labs

def minirocket_encode_dataset(model, data_tensor, batch_size=256):
    loader = DataLoader(data_tensor, batch_size=batch_size)
    reps, labs = [], []

    with torch.no_grad():
        for batch in loader:
            data, labels = batch
            data = data.view(data.size(0), -1, data.size(1))
            
            # MiniRocket expects NumPy input
            X_np = data.numpy()

            # Fit once (only if not already fitted)
            if not hasattr(model, "transformer_"):
                model.fit(X_np)

            # Transform returns NumPy or DataFrame
            rep = model.transform(X_np)
            if isinstance(rep, pd.DataFrame):
                rep = rep.values  # convert DataFrame → NumPy array

            # Convert to torch tensor
            rep = torch.tensor(rep, dtype=torch.float32)

            reps.append(rep)
            labs.append(labels)

    # Concatenate all batches
    return torch.cat(reps, dim=0), torch.cat(labs, dim=0)

def raw_signal_evaluation(train_dataset, valid_dataset, eval_protocol='linear'):

    train_repr, train_labels = encode_raw_dataset(train_dataset, batch_size=256)
    test_repr, test_labels  = encode_raw_dataset(valid_dataset, batch_size=256)


    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    elif eval_protocol == 'random_forest':
        fit_clf = eval_protocols.fit_rf
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)


    clf = fit_clf(train_repr, train_labels)

    val_acc = clf.score(test_repr, test_labels)
    
    predicted = clf.predict(test_repr)

    precision = precision_score(test_labels, predicted, average='macro', zero_division=0)
    recall = recall_score(test_labels, predicted, average='macro', zero_division=0)

    f1 = f1_score(test_labels, predicted, average='macro', zero_division=0)
    
    return [val_acc*100, f1, precision, recall]

def minirocket_signal_evaluation(train_dataset, valid_dataset, eval_protocol='linear'):

    if train_dataset[0][0].shape[-1] > 1:
        minirocket = MiniRocketMultivariate()
    else:
        minirocket = MiniRocket()


    train_repr, train_labels = minirocket_encode_dataset(minirocket, train_dataset, batch_size=256)
    test_repr, test_labels  = minirocket_encode_dataset(minirocket, valid_dataset, batch_size=256)


    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_rf
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    elif eval_protocol == 'random_forest':
        fit_clf = eval_protocols.fit_rf
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)


    clf = fit_clf(train_repr, train_labels)

    val_acc = clf.score(test_repr, test_labels)
    
    predicted = clf.predict(test_repr)

    precision = precision_score(test_labels, predicted, average='macro', zero_division=0)
    recall = recall_score(test_labels, predicted, average='macro', zero_division=0)

    f1 = f1_score(test_labels, predicted, average='macro', zero_division=0)
    
    return [val_acc*100, f1, precision, recall]
