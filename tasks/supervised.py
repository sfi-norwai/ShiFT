import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn.functional as F






import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

def supervised_evaluation(model, train_dataset, valid_dataset, feature_dim, num_epochs, batch_size, config, eval_protocol='linear'):

    train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=len(train_dataset),
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = False,
            )
            
    valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=len(valid_dataset),
                shuffle = False,
                num_workers=config.NUM_WORKERS,
                drop_last = False,
            )

    for batch in train_loader:
        train_data, train_labels = batch

        
    
    for batch2 in valid_loader:
        test_data, test_labels = batch2


    train_rep = model.encode(train_data)
    test_rep = model.encode(test_data)

    train_repr_pool = F.max_pool1d(
                    train_rep.transpose(1, 2),
                    kernel_size = train_rep.size(1),
                ).transpose(1, 2)
    
    train_repr_pool = train_repr_pool.squeeze(1)

    
    test_repr_pool = F.max_pool1d(
                    test_rep.transpose(1, 2),
                    kernel_size = test_rep.size(1),
                ).transpose(1, 2)
    
    test_repr_pool = test_repr_pool.squeeze(1)

    train_repr = train_repr_pool.detach().numpy()
    train_labels = train_labels.detach().numpy()

    test_repr = test_repr_pool.detach().numpy()
    test_labels = test_labels.detach().numpy()

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