import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def encode_dataset2(model, dataset, batch_size=256, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = False)
    reps = []
    labs = []
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            rep = model.encode(data)          # (B, C, T)
            rep = rep.mean(dim=1)            # (B, C) fast pooling

            reps.append(rep.cpu())
            labs.append(labels.cpu())

    reps = torch.cat(reps, dim=0).numpy()
    labs = torch.cat(labs, dim=0).numpy()

    return reps, labs

def encode_dataset(model, data_tensor, batch_size=256, device='cpu'):

    loader = DataLoader(data_tensor, batch_size=batch_size, shuffle = False)
    reps = []
    labs = []
    with torch.no_grad():
        for batch in loader:

            data, labels = batch
            data = data.to(device)
            rep = model.encode(data)  # [batch, seq_len, embed_dim]
            rep = rep.squeeze(1)      # remove seq_len if = 1

            reps.append(rep.cpu())    # move to CPU
            labs.append(labels.cpu())    # move to CPU

    
    return torch.cat(reps, dim=0), torch.cat(labs, dim=0)

def encode_datasetpfine(model, data_tensor, batch_size=256, device='cpu'):

    loader = DataLoader(data_tensor, batch_size=batch_size)
    reps = []
    labs = []
    with torch.no_grad():
        for batch in loader:

            data, labels = batch
            data = data.to(device)
            data = data.view(data.size(0), data.size(1), -1)
            rep = model(data)  # [batch, seq_len, embed_dim]
            rep = rep.squeeze(1)      # remove seq_len if = 1

            reps.append(rep.cpu())    # move to CPU
            labs.append(labels.cpu())    # move to CPU

    
    return torch.cat(reps, dim=0), torch.cat(labs, dim=0)

def supervised_evaluation_pfine(model, train_dataset, valid_dataset, feature_dim, num_epochs, batch_size, config, eval_protocol='linear'):

    train_repr1, train_labels = encode_datasetpfine(model.feature_extractor, train_dataset, batch_size=256)
    test_repr1, test_labels  = encode_datasetpfine(model.feature_extractor, valid_dataset, batch_size=256)

    train_repr = F.max_pool1d(
                    train_repr1.transpose(1, 2),
                    kernel_size = train_repr1.size(1),
                ).transpose(1, 2).squeeze(1).numpy()
    
    test_repr = F.max_pool1d(
                    test_repr1.transpose(1, 2),
                    kernel_size = test_repr1.size(1),
                ).transpose(1, 2).squeeze(1).numpy()

    # train_repr = train_repr_pool.view(-1, train_repr1.size(2)).numpy()
    # train_labels = train_labels1.view(-1).cpu().numpy()

    # test_repr = test_repr1.view(-1, test_repr1.size(2)).numpy()
    # test_labels = test_labels1.view(-1).cpu().numpy()

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
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
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)


    predicted = clf.predict(test_repr)

    precision = precision_score(test_labels, predicted, average='macro', zero_division=0)
    recall = recall_score(test_labels, predicted, average='macro', zero_division=0)

    f1 = f1_score(test_labels, predicted, average='weighted', zero_division=0)
    
    
    return {f"Val Accuracy: {val_acc*100:.2f}%, F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUPRC: {auprc:.2f}"}

def supervised_evaluation_ponly(model, train_dataset, valid_dataset, feature_dim, num_epochs, batch_size, config, eval_protocol='linear'):

    train_repr, train_labels = encode_dataset2(model, train_dataset, batch_size=256)
    test_repr, test_labels  = encode_dataset2(model, valid_dataset, batch_size=256)

    # train_repr = train_repr_pool.view(-1, train_repr1.size(2)).numpy()
    # train_labels = train_labels1.view(-1).cpu().numpy()

    # test_repr = test_repr1.view(-1, test_repr1.size(2)).numpy()
    # test_labels = test_labels1.view(-1).cpu().numpy()

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
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
    #if eval_protocol == 'linear':
    #    y_score = clf.predict_proba(test_repr)
    #else:
    #    y_score = clf.decision_function(test_repr)
    #test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))

    # auprc = average_precision_score(test_labels_onehot, y_score)


    predicted = clf.predict(test_repr)

    precision = precision_score(test_labels, predicted, average='macro', zero_division=0)
    recall = recall_score(test_labels, predicted, average='macro', zero_division=0)

    f1 = f1_score(test_labels, predicted, average='macro', zero_division=0)
    
    
    # return {f"Val Accuracy: {val_acc*100:.2f}%, F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUPRC: {auprc:.2f}"}
    return [val_acc*100, f1, precision, recall]

def supervised_evaluation_ponly_block(model, train_dataset, valid_dataset, feature_dim, num_epochs, batch_size, config, eval_protocol='linear'):

    train_repr, train_labels = encode_dataset(model, train_dataset, batch_size=256)
    test_repr, test_labels  = encode_dataset(model, valid_dataset, batch_size=256)


    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
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
    #if eval_protocol == 'linear':
    #    y_score = clf.predict_proba(test_repr)
    #else:
    #    y_score = clf.decision_function(test_repr)
    #test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))

    # auprc = average_precision_score(test_labels_onehot, y_score)


    predicted = clf.predict(test_repr)

    precision = precision_score(test_labels, predicted, average='macro', zero_division=0)
    recall = recall_score(test_labels, predicted, average='macro', zero_division=0)

    f1 = f1_score(test_labels, predicted, average='macro', zero_division=0)
    
    
    # return {f"Val Accuracy: {val_acc*100:.2f}%, F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUPRC: {auprc:.2f}"}
    return [val_acc*100, f1, precision, recall]



def clustering_evaluation_ponly(model, valid_dataset, seed, eval_protocol='kmeans'):

    test_repr, test_labels  = encode_dataset2(model, valid_dataset, batch_size=256)


    test_repr = StandardScaler().fit_transform(test_repr)
    n_clusters = len(np.unique(test_labels))


    if eval_protocol == 'kmeans':
        clt_type = KMeans(
                n_clusters=n_clusters,
                n_init=1,
                random_state=seed
            )
    elif eval_protocol == 'dbscan':
        clt_type = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric="euclidean"
            )
    else:
        assert False, 'unknown clustering algorithm'

    cluster_ids = clt_type.fit_predict(test_repr)

    nmi = normalized_mutual_info_score(test_labels, cluster_ids)
    ari = adjusted_rand_score(test_labels, cluster_ids)

    return [nmi, ari]

def clustering_evaluation_ponly_block(model, valid_dataset, seed, eval_protocol='kmeans'):

    test_repr, test_labels  = encode_dataset(model, valid_dataset, batch_size=256)


    test_repr = StandardScaler().fit_transform(test_repr)
    n_clusters = len(np.unique(test_labels))


    if eval_protocol == 'kmeans':
        clt_type = KMeans(
                n_clusters=n_clusters,
                n_init=1,
                random_state=seed
            )
    elif eval_protocol == 'dbscan':
        clt_type = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric="euclidean"
            )
    else:
        assert False, 'unknown clustering algorithm'

    cluster_ids = clt_type.fit_predict(test_repr)
    
    nmi = normalized_mutual_info_score(test_labels, cluster_ids)
    ari = adjusted_rand_score(test_labels, cluster_ids)

    return [nmi, ari]
