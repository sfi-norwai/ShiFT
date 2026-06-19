import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import pickle
import src.config, src.utils
import einops
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def extract_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y)
    return np.array(labels)

def stratified_percentage_ecg2(dataset, percentage, seed=0):
    rng = np.random.default_rng(seed)

    labels = extract_labels(dataset)
    unique_labels = np.unique(labels)

    selected = []

    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        k = max(1, int(len(idxs) * percentage))
        chosen = rng.choice(idxs, size=k, replace=False)
        selected.extend(chosen)

    return Subset(dataset, selected)

def split_dataset(data, label, validation_ratio):
    
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label

def stratified_percentage(dataset, percentage, random_state=42):
    """
    Selects a fixed percentage of samples from each class in a stratified way.

    Args:
        dataset: a dataset with attribute dataset.y_data containing labels
        percentage: float in (0,1], e.g. 0.2 for 20%
    """
    labels = np.array(dataset.y_data)
    rng = np.random.default_rng(random_state)

    # collect indices for each class
    class_indices = {lbl: np.where(labels == lbl)[0] for lbl in np.unique(labels)}

    selected_indices = []
    for lbl, indices in class_indices.items():
        n_samples = max(1, int(len(indices) * percentage))
        replace = len(indices) < n_samples  # if not enough, allow replacement
        chosen = rng.choice(indices, size=n_samples, replace=replace)
        selected_indices.extend(chosen)

    return Subset(dataset, selected_indices)

def stratified_fixed_count(dataset, n_per_class, random_state=42):
    
    labels = dataset.y_data
    rng = np.random.default_rng(random_state)
    labels = np.array(labels)

    # collect indices for each class
    class_indices = {lbl: np.where(labels == lbl)[0] for lbl in np.unique(labels)}

    selected_indices = []
    for lbl, indices in class_indices.items():
        replace = len(indices) < n_per_class  # allow replacement if too few
        chosen = rng.choice(indices, size=n_per_class, replace=replace)
        selected_indices.extend(chosen)

    return Subset(dataset, selected_indices)

def stratified_fixed_count_ecg2(dataset, n_per_class, random_state=42):
    rng = np.random.default_rng(random_state)

    labels = extract_labels(dataset)
    unique_labels = np.unique(labels)

    selected_indices = []

    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]

        # allow replacement if class is too small
        replace = len(indices) < n_per_class
        chosen = rng.choice(indices, size=n_per_class, replace=replace)

        selected_indices.extend(chosen.tolist())

    return Subset(dataset, selected_indices)

# Function to load a pickle file
def load_pickle_file(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def feature_stft(x_train, n_fft = 100, hop_length=50, win_length=100, phase = False, stack_axes = True):
   
    train_tensor = torch.tensor(x_train).transpose(1,2).reshape(-1, 2).transpose(0,1)

    x = torch.stft(
        input=train_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=False,
        return_complex=True)  # [num_channels, num_bins, num_frames]

    x_cartesian = src.utils.complex_to_cartesian(x)
    x_magnitude = src.utils.complex_to_magnitude(x, expand=True)

    x = x_cartesian if phase else x_magnitude
    if stack_axes:
        # Stack all spectrograms and put time dim first:
        # [num_channels, num_bins, num_frames, stft_parts] ->
        # [num_frames, num_channels x num_bins x stft_parts]
        x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
    else:
        x = einops.rearrange(x, 'C F T P -> T C F P')
    
    return x

def feature_sleep(x_train, n_fft = 100, hop_length=50, win_length=100, phase = False, stack_axes = True):
    
    train_tensor = x_train.squeeze(1).reshape(1, -1)
    
    x = torch.stft(
        input=train_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=False,
        return_complex=True)  # [num_channels, num_bins, num_frames]

    x_cartesian = src.utils.complex_to_cartesian(x)
    x_magnitude = src.utils.complex_to_magnitude(x, expand=True)

    x = x_cartesian if phase else x_magnitude
    if stack_axes:
        # Stack all spectrograms and put time dim first:
        # [num_channels, num_bins, num_frames, stft_parts] ->
        # [num_frames, num_channels x num_bins x stft_parts]
        x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
    else:
        x = einops.rearrange(x, 'C F T P -> T C F P')
    
    return x

def feature_kpi(x_train, n_fft = 10, hop_length=5, win_length=10, phase = False, stack_axes = True):
    
    
   
    train_tensor = torch.tensor(x_train).unsqueeze(1).reshape(1, -1)
    
    x = torch.stft(
        input=train_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=False,
        return_complex=True)  # [num_channels, num_bins, num_frames]

    x_cartesian = src.utils.complex_to_cartesian(x)
    x_magnitude = src.utils.complex_to_magnitude(x, expand=True)

    x = x_cartesian if phase else x_magnitude
    if stack_axes:
        # Stack all spectrograms and put time dim first:
        # [num_channels, num_bins, num_frames, stft_parts] ->
        # [num_frames, num_channels x num_bins x stft_parts]
        x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
    else:
        x = einops.rearrange(x, 'C F T P -> T C F P')
    
    return x

def windowed_labels(
    labels,
    num_labels,
    frame_length,
    frame_step=None,
    pad_end=False,
    kind='density',
):
    """Generates labels that correspond to STFTs

    With kind=None we are able to split the given labels
    array into batches. (T, C) -> (B, T', C)

    Parameters
    ----------
    labels : np.array

    Returns
    -------
    np.array
    """
    labels = torch.tensor(labels).view(-1)
    
    # Labels should be a single vector (int-likes) or kind has to be None
    labels = np.asarray(labels)
    
    if kind is not None and not labels.ndim == 1:
        raise ValueError('Labels must be a vector')
    if not (labels >= 0).all():
        raise ValueError('All labels must be >= 0')
    if not (labels < num_labels).all():
        raise ValueError(f'All labels must be < {num_labels} (num_labels)')
    # Kind determines how labels in each window should be processed
    if not kind in {'counts', 'density', 'onehot', 'argmax', None}:
        raise ValueError('`kind` must be in {counts, density, onehot, argmax, None}')
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process labels with a sliding window. TODO: vectorize?
    output = []
    for i in range(0, len(labels), frame_step):
        chunk = labels[i:i+frame_length]
        chunk = chunk.astype(int)
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        # Just append the chunk if kind is None
        if kind == None:
            output.append(chunk)
            continue
        # Count the occurences of each label
        counts = np.bincount(chunk, minlength=num_labels)
        # Then process based on kind
        if kind == 'counts':
            output.append(counts)
        elif kind == 'density':
            output.append(counts / len(chunk))
        elif kind == 'onehot':
            one_hot = np.zeros(num_labels)
            one_hot[np.argmax(counts)] = 1
            output.append(one_hot)
        elif kind == 'argmax':
            output.append(np.argmax(counts))
    if pad_end:
        return output
    else:
        return torch.tensor(output)
    
class STFTDataset(Dataset):
    
    def __init__(self, data_path, class_to_exclude=3, n_fft = 250, hop_length=125, win_length=250, seq_length=500, num_labels=4):
        """
        Args:
            x_data (Tensor): The input features, e.g., from STFT.
            y_data (Tensor): The corresponding labels, windowed and processed.
            seq_length (int): The length of each sequence.
        """
        
        # Load each of the pickle files
        tensor_data = load_pickle_file(f'./{data_path}/tensor_data.pkl')
        tensor_label = load_pickle_file(f'./{data_path}/tensor_label.pkl')
        
        x_data = feature_stft(tensor_data, n_fft = n_fft, hop_length=hop_length, win_length=win_length)
        y_data = windowed_labels(labels=tensor_label, num_labels=num_labels, frame_length=n_fft, frame_step=hop_length, kind='argmax')

        self.x_data = x_data
        self.y_data = y_data
        self.seq_length = seq_length
        # self.class_to_exclude = class_to_exclude
        
        # Create a mask that filters out the class_to_exclude
        # mask = self.y_data != self.class_to_exclude

        # Apply the mask to filter the data
        # self.x_data = self.x_data[mask]
        # self.y_data = self.y_data[mask]
        
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]

        return x_seq, y_seq

class SequentialRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size


    def __iter__(self):
        

        indices = list(range(len(self.data_source)))
        
        remaining = len(indices) % self.batch_size
        if remaining > 0:
            indices = indices[:-remaining]
        final_indices = np.reshape(indices, (-1, self.batch_size))

        # Shuffle the batches
        np.random.shuffle(final_indices)

        # Flatten the list of batches to get the final order of indices
        final_indices = [idx for batch in final_indices for idx in batch]
        
        return iter(final_indices)

    def __len__(self):
        return len(self.data_source)
    

# Wrapper dataset class to flatten the batches
class FlattenedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_batches = len(original_dataset)
        self.batch_size = original_dataset[0][0].shape[0]  # Assuming shape [599, 156]

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        sample_idx = idx % self.batch_size
        data_batch, label_batch = self.original_dataset[batch_idx]
        
        return data_batch[sample_idx], label_batch[sample_idx]
    
# class SLEEPDataset(Dataset):
    
#     def __init__(self, data_path, n_fft = 178, hop_length=89, win_length=178, seq_length=119, num_labels=5, eval=False):
#         """
#         Args:
#             x_data (Tensor): The input features, e.g., from STFT.
#             y_data (Tensor): The corresponding labels, windowed and processed.
#             seq_length (int): The length of each sequence.
#         """
        
#         # Load each of the pickle files
#         if eval:
#             data = torch.load(f'{data_path}/val.pt')
#         else:
#             data = torch.load(f'{data_path}/train.pt')

#         x_data = data['samples'].squeeze()
#         self.y_data = data['labels']

#         self.x_data = feature_sleep(x_data, n_fft = n_fft, hop_length=hop_length, win_length=win_length)
#         self.seq_length = seq_length
        
#     def __len__(self):
#         # Return the number of full sequences in the dataset
#         return len(self.x_data) // self.seq_length

#     def __getitem__(self, idx):
#         """
#         Returns a tuple (input, label) for the given index.
#         The input is reshaped to (seq_length, features).
#         """
#         start_idx = idx * self.seq_length
#         end_idx = start_idx + self.seq_length

#         # Extract the sequence of data and corresponding labels
#         x_seq = self.x_data[start_idx:end_idx]
#         y_seq = self.y_data[start_idx:end_idx]

#         return x_seq, y_seq
    

class PAMAP2Dataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/PAMAP2-001.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq
    
class SKODADataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/Skoda.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq
    
class USCHADDataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/USC_HAD.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq
    
    
class OPPDataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/Opportunity.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq

class WISDMDataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/WISDM.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
            
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq
       
class WISDM2Dataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/WISDM2.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq
    
class HARDataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/HAR.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq
    
class EpilepsyDataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/Epilepsy.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq
    
class SleepmDataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        data_all = np.load(f'{data_path}/Sleep.npy', allow_pickle=True)
        data_dict = data_all.item()

        # Load each of the pickle files
        if eval:
            self.x_data = data_dict['test_data']
            self.y_data = data_dict['test_label']
        else:
            self.x_data = data_dict['train_data']
            self.y_data = data_dict['train_label']


     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq

class KpiDataset(Dataset):

    def __init__(self, data_path, n_fft = 250, hop_length=125, win_length=250, seq_length=500, num_labels=2):
        """
        Args:
            x_data (Tensor): The input features, e.g., from STFT.
            y_data (Tensor): The corresponding labels, windowed and processed.
            seq_length (int): The length of each sequence.
        """
        
        
        df = pd.read_csv(f'{data_path}/train.csv')
        tensor_data = df['value'].values 
        tensor_label = df['label'].values  
        
        x_data = feature_kpi(tensor_data, n_fft = n_fft, hop_length=hop_length, win_length=win_length)
        y_data = windowed_labels(labels=tensor_label, num_labels=num_labels, frame_length=n_fft, frame_step=hop_length, kind='argmax')

        self.x_data = x_data
        self.y_data = y_data
        self.seq_length = seq_length
        
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]

        return x_seq, y_seq
    
class PAMAPDatasetNP(Dataset):
    def __init__(self, data_path, eval, seq_len=20):
        data = np.load(f'{data_path}/PAMAP2-001.npy', allow_pickle=True).item()
        
        if eval:
            self.x_data = data['test_data']
            self.y_data = data['test_label']
        else:
            self.x_data = data['train_data']
            self.y_data = data['train_label']
            
        self.seq_len = seq_len

        # Number of full non-overlapping sequences
        self.num_sequences = len(self.x_data) // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        x_seq = self.x_data[start_idx:end_idx]  # shape: [seq_len, 100, 52]
        y_seq = self.y_data[start_idx:end_idx]  # keep as sequence (seq-to-seq)
        
        return torch.tensor(x_seq.transpose(0, 2, 1), dtype=torch.float32), torch.tensor(y_seq, dtype=torch.long)
    
class SleepDataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        # Load each of the pickle files
        if eval:
            data = torch.load(f'{data_path}/test.pt')
            self.x_data = data['samples']
            self.y_data = data['labels']
        else:
            data = torch.load(f'{data_path}/train.pt')
            self.x_data = data['samples']
            self.y_data = data['labels']
     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """
        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq
    

class ECGDataset3(Dataset):
    def __init__(self, data_path, eval=False, window_size=2500, step_size=2500):
        """
        ECG dataset loader that splits data by subject.
        
        Args:
            data_path (str): Directory containing tensor_data.pkl and tensor_label.pkl
            eval (bool): If True, load test (held-out subjects)
            window_size (int): Length of each segment
            step_size (int): Stride between segments
        """

        # Load pre-saved pickles
        tensor_data = load_pickle_file(f'{data_path}/tensor_data.pkl')  # shape: (23, 2, T)
        tensor_label = load_pickle_file(f'{data_path}/tensor_label.pkl')  # shape: (23, T)

        tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
        tensor_label = torch.tensor(tensor_label, dtype=torch.long)

        # Subject-based split
        num_subjects = tensor_data.shape[0]
        assert num_subjects >= 23, "Expected at least 23 subjects."

        train_subjects = list(range(5, 23))
        test_subjects = list(range(0, 5))

        if eval:
            subject_indices = test_subjects
        else:
            subject_indices = train_subjects

        x_segments, y_segments = [], []

        # Create windowed samples for the selected subjects
        for i in subject_indices:
            x = tensor_data[i]      # (2, T)
            y = tensor_label[i]     # (T,)
            T = x.shape[1]

            # Sliding window segmentation
            for start in range(0, T - window_size + 1, step_size):
                end = start + window_size
                x_win = x[:, start:end]
                y_win = y[start:end].mode().values.item()  # majority label in window
                x_segments.append(x_win)
                y_segments.append(y_win)

        # Stack all windows
        self.x_data = torch.stack(x_segments)   # (num_windows, 2, window_size)
        self.y_data = torch.tensor(y_segments)  # (num_windows,)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # (seq_len, channels) format for models like RNNs or Transformers
        x_seq = self.x_data[idx].transpose(1, 0)
        y_seq = self.y_data[idx]
        return x_seq, y_seq
    
class HARTHDataset(Dataset):
    def __init__(self, data_path, eval=False, window_size=500, step_size=250,
                 sep=',', header=0, y_column='label',
                 x_columns=None, drop_labels=None,
                 source_freq=None, target_freq=None, label_map=None):
        """
        HARTH dataset loader that splits by subject (22 subjects total).

        Args:
            data_path (str): Directory containing subject CSV files (S001.csv ... S022.csv)
            eval (bool): If True, loads test subjects (last 4 by default)
            window_size (int): Length of each window in samples
            step_size (int): Step size between consecutive windows
            sep (str): CSV separator
            header (int): CSV header row index
            y_column (str): Label column name
            x_columns (list): List of feature columns to use
            drop_labels (list): Optional list of labels to drop
            source_freq (float): Original sampling frequency
            target_freq (float): Target resampling frequency (if resampling)
            label_map (dict): Optional mapping for labels
        """
        self.sep = sep
        self.header = header
        self.y_column = y_column
        self.drop_labels = drop_labels or []
        self.source_freq = source_freq
        self.target_freq = target_freq
        self.x_columns = x_columns

        # --- Get subject files ---
        csv_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
        num_subjects = len(csv_files)
        print(f"Found {num_subjects} subjects")

        assert num_subjects == 22, f"Expected 22 subjects in HARTH dataset, found {num_subjects}."

        # --- Split train/test subjects ---
        train_subjects = csv_files[:18]
        test_subjects = csv_files[18:]
        selected_files = test_subjects if eval else train_subjects

        x_segments, y_segments = [], []
        all_labels = set()

        # --- First pass: gather all unique labels ---
        for fname in selected_files:
            df = pd.read_csv(os.path.join(data_path, fname), sep=self.sep, header=self.header)
            all_labels.update(df[self.y_column].unique())

        # --- Build label map if not provided ---
        if label_map is None:
            sorted_labels = sorted(list(all_labels))
            label_map = {old: new for new, old in enumerate(sorted_labels)}
        self.label_map = label_map

        # --- Load each subject ---
        for fname in tqdm(selected_files, desc="Loading subjects"):
            fpath = os.path.join(data_path, fname)
            df = pd.read_csv(fpath, sep=self.sep, header=self.header)

            # Infer feature columns if not provided
            if self.x_columns is None:
                self.x_columns = [c for c in df.columns if c not in [self.y_column, 'timestamp']]

            # Drop unwanted labels if any
            if self.drop_labels:
                df = df[~df[self.y_column].isin(self.drop_labels)]

            df = df.dropna()

            # Apply label remapping
            df[self.y_column] = df[self.y_column].map(self.label_map)

            # --- Convert to tensors ---
            x = torch.tensor(df[self.x_columns].values.T, dtype=torch.float32)  # (C, T)
            y = torch.tensor(df[self.y_column].values, dtype=torch.long)        # (T,)

            # --- Segment into overlapping windows ---
            for start in range(0, x.shape[1] - window_size + 1, step_size):
                end = start + window_size
                x_win = x[:, start:end]
                y_win = y[start:end].mode().values.item()  # majority label
                x_segments.append(x_win)
                y_segments.append(y_win)

        self.x_data = torch.stack(x_segments)  # (N, C, W)
        self.y_data = torch.tensor(y_segments)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_seq = self.x_data[idx].transpose(1, 0)  # (W, C)
        y_seq = self.y_data[idx]
        return x_seq, y_seq
    

class FeatureSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, feature_idx):
        self.base_dataset = base_dataset
        self.feature_idx = feature_idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        x = x[:, self.feature_idx]   # (T, 3)
        return x, y

class ECGDataset(Dataset):
    
    def __init__(self, data_path, eval=False):

        # Load each of the pickle files
        if eval:
            data = torch.load(f'{data_path}/test.pt')
            self.x_data = data['samples']
            self.y_data = data['labels']
        else:
            data = torch.load(f'{data_path}/train.pt')
            
            self.x_data = data['samples']
            self.y_data = data['labels']
     
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """
        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq

class ECGDataset2(Dataset):
    def __init__(self, data_path, eval=False, num_long_windows=72, sample_window=2500):
        """
        TNC-style ECG dataset. Each subject is split into a few (e.g., 5) long segments,
        and at __getitem__ a random window of length sample_window is drawn.
        
        Args:
            data_path (str): dir containing tensor_data.pkl (23, 2, T), tensor_label.pkl (23, T)
            eval (bool): choose held-out subjects
            num_long_windows (int): number of large splits per subject (default 5)
            sample_window (int): length of sampled window returned by __getitem__
        """

        # Load raw data
        tensor_data = load_pickle_file(f'{data_path}/tensor_data.pkl')   # (23, 2, T)
        tensor_label = load_pickle_file(f'{data_path}/tensor_label.pkl') # (23, T)

        tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
        tensor_label = torch.tensor(tensor_label, dtype=torch.long)

        num_subjects = tensor_data.shape[0]
        assert num_subjects >= 23

        # Subject split (held-out 5 subjects)
        train_subjects = list(range(5, 23))
        test_subjects  = list(range(0, 5))
        subject_indices = test_subjects if eval else train_subjects

        self.sample_window = sample_window
        self.long_chunks = []  # (x_chunk, y_chunk)

        # ---- Create long windows ----
        for sid in subject_indices:
            x = tensor_data[sid]   # (2, T)
            y = tensor_label[sid]  # (T,)
            T = x.shape[1]

            chunk_len = T // num_long_windows

            for k in range(num_long_windows):
                start = k * chunk_len
                end   = (k + 1) * chunk_len if k < num_long_windows - 1 else T

                x_chunk = x[:, start:end]   # (2, L)
                y_chunk = y[start:end]      # (L,)

                # keep only chunks long enough to sample windows from
                if x_chunk.shape[1] >= sample_window:
                    self.long_chunks.append((x_chunk, y_chunk))

        # Now dataset length = num_subjects × num_long_windows  
        # Example: 18 subjects * 5 chunks = 90 items for training
        print(f"Created {len(self.long_chunks)} long windows")

    def __len__(self):
        return len(self.long_chunks)

    def __getitem__(self, idx):
        """
        Returns a RANDOM window of length sample_window from the chosen long chunk.
        """

        x_long, y_long = self.long_chunks[idx]     # shapes (2, L) and (L,)
        L = x_long.shape[1]

        # Random starting index
        start = torch.randint(0, L - self.sample_window + 1, (1,)).item()
        end   = start + self.sample_window

        # Extract slice
        x_win = x_long[:, start:end]        # (2, window)
        y_win = y_long[start:end]

        # majority class of slice
        y_label = y_win.mode().values.item()

        # Return format: (window, channels)
        return x_win.transpose(1, 0), y_label

class ECGNormDataset(Dataset):

    def __init__(self, data_path, window_size=2500, step_size=2500, eval=False):

        # Load each of the pickle files
        if eval:
            x_data = load_pickle_file(f'{data_path}/x_test.pkl')   # (5, 2, T)
            self.x_data = torch.tensor(x_data, dtype=torch.float32)

            y_data = load_pickle_file(f'{data_path}/state_test.pkl')   # (5, T)
            self.y_data = torch.tensor(y_data, dtype=torch.long)
        else:
            x_data = load_pickle_file(f'{data_path}/x_train.pkl')   # (18, 2, T)
            self.x_data = torch.tensor(x_data, dtype=torch.float32)

            y_data = load_pickle_file(f'{data_path}/state_train.pkl')   # (18, T)
            self.y_data = torch.tensor(y_data, dtype=torch.long)

        x_segments, y_segments = [], []

        for i in range(len(self.x_data)):
            x = self.x_data[i]      # (2, T)
            y = self.y_data[i]     # (T,)
            T = x.shape[1]

            # Sliding window segmentation
            for start in range(0, T - window_size + 1, step_size):
                end = start + window_size
                x_win = x[:, start:end]
                y_win = y[start:end].mode().values.item()  # majority label in window
                x_segments.append(x_win)
                y_segments.append(y_win)

        # Stack all windows
        self.x_data = torch.stack(x_segments)   # (num_windows, 2, window_size)
        self.y_data = torch.tensor(y_segments)  # (num_windows,)

    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx].transpose(1,0)
        y_seq = self.y_data[idx]

        return x_seq, y_seq



class ECGDatasetDeterministic(Dataset):
    def __init__(self, data_path, eval=False, num_long_windows=72, sample_window=2500):
        """
        Deterministic TNC-style ECG dataset. Each subject is split into long segments,
        and __getitem__ returns a fixed window from the middle of each chunk.

        Args:
            data_path (str): directory containing tensor_data.pkl (23, 2, T) and tensor_label.pkl (23, T)
            eval (bool): choose held-out subjects
            num_long_windows (int): number of large splits per subject
            sample_window (int): length of window returned
        """

        # Load raw data
        tensor_data = load_pickle_file(f'{data_path}/tensor_data.pkl')   # (23, 2, T)
        tensor_label = load_pickle_file(f'{data_path}/tensor_label.pkl') # (23, T)

        tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
        tensor_label = torch.tensor(tensor_label, dtype=torch.long)

        num_subjects = tensor_data.shape[0]
        assert num_subjects >= 23

        # Subject split (held-out 5 subjects)
        train_subjects = list(range(5, 23))
        test_subjects  = list(range(0, 5))
        subject_indices = test_subjects if eval else train_subjects

        self.sample_window = sample_window
        self.long_chunks = []  # (x_chunk, y_chunk)

        # ---- Create long windows ----
        for sid in subject_indices:
            x = tensor_data[sid]   # (2, T)
            y = tensor_label[sid]  # (T,)
            T = x.shape[1]

            chunk_len = T // num_long_windows

            for k in range(num_long_windows):
                start = k * chunk_len
                end   = (k + 1) * chunk_len if k < num_long_windows - 1 else T

                x_chunk = x[:, start:end]   # (2, L)
                y_chunk = y[start:end]      # (L,)

                # keep only chunks long enough to sample windows from
                if x_chunk.shape[1] >= sample_window:
                    self.long_chunks.append((x_chunk, y_chunk))

        print(f"Created {len(self.long_chunks)} long windows")

    def __len__(self):
        return len(self.long_chunks)

    def __getitem__(self, idx):
        """
        Returns a deterministic window of length sample_window from the middle of the chunk.
        """
        x_long, y_long = self.long_chunks[idx]  # shapes (2, L) and (L,)
        L = x_long.shape[1]

        # Middle index
        mid = L // 2
        start = max(0, mid - self.sample_window // 2)
        end = start + self.sample_window

        # Extract slice
        x_win = x_long[:, start:end]  # (2, window)
        y_win = y_long[start:end]

        # majority class of slice
        y_label = y_win.mode().values.item()

        # Return format: (window, channels)
        return x_win.transpose(1, 0), y_label

class TNCDatasetFromDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mc_sample_size, window_size, augmentation=1, epsilon=3, state=None, adf=False):
        super().__init__()
        self.dataset = dataset         # your existing dataset
        self.window_size = window_size
        self.augmentation = augmentation
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.adf = adf
        self.epsilon = epsilon
        self.delta = 5 * window_size * epsilon

        # Precompute T for each sample
        self.T_list = [len(dataset[i][0]) if isinstance(dataset[i], tuple) else len(dataset[i])
                       for i in range(len(dataset))]

    def __len__(self):
        return len(self.dataset) * self.augmentation

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)

        # get the sample from original dataset
        x = self.dataset[idx]

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if isinstance(x, tuple):
            x = x[0]  # if dataset returns (data, label)

        # x = x.float()
        x = x.transpose(1,0)
        T = x.shape[-1]

        # pick random center
        t = np.random.randint(2 * self.window_size, T - 2 * self.window_size)
        x_t = x[:, t - self.window_size//2 : t + self.window_size//2]

        X_close = self._find_neighbours(x, t, T)
        X_distant = self._find_non_neighbours(x, t, T)

        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[idx][t-self.window_size//2:t+self.window_size//2]))

        return x_t, X_close, X_distant, y_t

    def _find_neighbours(self, x, t, T):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.long)

        t_p = [int(t + np.random.randn() * self.epsilon * self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2 + 1, min(tp, T - self.window_size//2)) for tp in t_p]
        x_p = torch.stack([x[:, tp-self.window_size//2:tp+self.window_size//2] for tp in t_p])
        return x_p

    def _find_non_neighbours(self, x, t, T):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.long)
        if t > T/2:
            t_n = np.random.randint(self.window_size//2, max(t - self.delta + 1, self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min(t + self.delta, T - self.window_size-1), T - self.window_size//2, self.mc_sample_size)
        x_n = torch.stack([x[:, tn-self.window_size//2:tn+self.window_size//2] for tn in t_n])
        return x_n

class SubwindowDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset, n=5, p=0.5):
        self.base = base_dataset
        self.n = n
        self.p = p

    def compute_params(self, window):
        # theoretical L
        L = window / (1 + (self.n - 1) * (1 - self.p))
        L = int(round(L / 2) * 2)       # force to even

        s = int(round(L * (1 - self.p)))  # stride
        return L, s

    def __getitem__(self, idx):
        x, y = self.base[idx]   # x: [window, feat]
        x = torch.Tensor(x)

        window, feat = x.shape
        L, s = self.compute_params(window)

        # number of subwindows
        n_sub = (window - L) // s + 1

        # as_strided extraction
        subwins = x.as_strided(
            size=(n_sub, L, feat),
            stride=(s * x.stride(0), x.stride(0), x.stride(1))
        ).contiguous()

        # return (subwindows, label)
        return subwins, y

    def __len__(self):
        return len(self.base)


import torch
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe, load_from_tsfile
from sklearn.preprocessing import LabelEncoder


def load_uea_ts_to_tensor(train_path, test_path):
    """
    Load UEA .ts files and return PyTorch tensors:
        X_train_tensor: [num_samples, num_channels, max_seq_len]
        y_train_tensor: [num_samples]  (encoded as integers)
        X_test_tensor:  [num_samples, num_channels, max_seq_len]
        y_test_tensor:  [num_samples]  (encoded as integers)
    """
    # Load .ts files
    X_train, y_train = load_from_tsfile_to_dataframe(train_path, return_separate_X_and_y=True)
    X_test,  y_test  = load_from_tsfile_to_dataframe(test_path, return_separate_X_and_y=True)

    # Encode string labels to integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # Determine number of channels and max sequence length
    num_train, num_channels = X_train.shape
    num_test = X_test.shape[0]
    max_len = max(
        max(len(s) for s in row) for row in X_train.itertuples(index=False)
    )
    max_len_test = max(
        max(len(s) for s in row) for row in X_test.itertuples(index=False)
    )
    max_len = max(max_len, max_len_test)  # ensure test series fit

    # Convert to PyTorch tensors with zero-padding
    def df_to_tensor(df):
        tensor = torch.zeros((df.shape[0], num_channels, max_len), dtype=torch.float32)
        for i, row in enumerate(df.itertuples(index=False)):
            for j, s in enumerate(row):
                tensor[i, j, :len(s)] = torch.tensor(s.values, dtype=torch.float32)
        return tensor

    X_train_tensor = df_to_tensor(X_train).transpose(2,1)
    X_test_tensor  = df_to_tensor(X_test).transpose(2,1)

    # Convert encoded labels to tensor
    y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test_enc, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_UEA(dataset):

    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:

            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

class UEADataset(Dataset):

    def __init__(self, data_name, eval=False):

        train_path = f"datasets/Multivariate_ts/{data_name}/{data_name}_TRAIN.ts"
        test_path  = f"datasets/Multivariate_ts/{data_name}/{data_name}_TEST.ts"

        X_train, y_train, X_test, y_test = load_uea_ts_to_tensor(train_path, test_path)
        
        #X_train, y_train, X_test, y_test = load_UEA(data_name)

        # Load each of the pickle files
        if eval:
            self.x_data = X_test
            self.y_data = y_test
        else:
            self.x_data = X_train
            self.y_data = y_train



    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx]
        y_seq = self.y_data[idx]

        return x_seq, y_seq


class UCRDataset(Dataset):

    def __init__(self, data_name, eval=False):

        train_path = f"datasets/Univariate_ts/{data_name}/{data_name}_TRAIN.ts"
        test_path  = f"datasets/Univariate_ts/{data_name}/{data_name}_TEST.ts"

        X_train, y_train, X_test, y_test = load_uea_ts_to_tensor(train_path, test_path)

        #X_train, y_train, X_test, y_test = load_UEA(data_name)

        # Load each of the pickle files
        if eval:
            self.x_data = X_test
            self.y_data = y_test
        else:
            self.x_data = X_train
            self.y_data = y_train



    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx]
        y_seq = self.y_data[idx]

        return x_seq, y_seq

class Capture24Dataset(Dataset):

    def __init__(self, data_path, eval=False):

        # Load each of the pickle files
        self.x_data = np.load(f'{data_path}/capture_X.npy', allow_pickle=True)
        self.y_data = np.load(f'{data_path}/capture_Y.npy', allow_pickle=True)

    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[idx]
        y_seq = self.y_data[idx]

        return x_seq, y_seq

class CaTTv1Dataset(Dataset):
    def __init__(self, base_dataset, seq_length):
        self.base_dataset = base_dataset
        self.seq_length = seq_length

    def __len__(self):
        return len(self.base_dataset) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length

        xs = []
        ys = []

        for i in range(start, end):
            item = self.base_dataset[i]

            if isinstance(item, tuple):
                x, y = item
                xs.append(x)
                ys.append(y)
            else:
                xs.append(item)

        xs = [torch.as_tensor(x) for x in xs]
        x_seq = torch.stack(xs, dim=0)  # (seq_len, window, channel)

        if ys:
            ys = [torch.as_tensor(y) for y in ys]
            return x_seq, torch.stack(ys, dim=0)

        return x_seq

