import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import pickle
import src.config, src.utils
import einops
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
import os


def split_dataset(data, label, validation_ratio):
    
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label

def stratified_fixed_count_catt(dataset, n_per_class, random_state=42):
    """
    Stratified sampling with fixed number of samples per class.
    Works with datasets where y_data is either (N,) or (B, S).

    If x_data has 4 dimensions (B, S, W, F), it is flattened to (B*S, W, F)
    and y_data is flattened accordingly to (B*S,).
    """
    rng = np.random.default_rng(random_state)

    # --- Flatten if necessary ---
    x_data = dataset.x_data
    y_data = dataset.y_data

    if y_data.ndim == 2:  # shape (batch, seq_len)
        y_data = y_data.reshape(-1)
    if x_data.ndim == 4:  # shape (batch, seq_len, win, feat)
        x_data = x_data.reshape(-1, *x_data.shape[2:])

    # --- Update dataset in memory (optional, for downstream use) ---

    dataset.x_data = x_data
    dataset.y_data = y_data

    print(x_data.shape)
    print(y_data.shape)


    # --- Stratified selection ---
    labels = np.array(y_data)
    class_indices = {lbl: np.where(labels == lbl)[0] for lbl in np.unique(labels)}

    selected_indices = []
    for lbl, indices in class_indices.items():
        replace = len(indices) < n_per_class
        chosen = rng.choice(indices, size=n_per_class, replace=replace)
        selected_indices.extend(chosen)

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


     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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


     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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


     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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


     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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

        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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


     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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


     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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


     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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


     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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
            data = torch.load(f'{data_path}/val.pt')
            self.x_data = data['samples']
            self.y_data = data['labels']
        else:
            data = torch.load(f'{data_path}/train.pt')
            self.x_data = data['samples']
            self.y_data = data['labels']
     
        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
        return x_seq, y_seq
    
class ECGDataset(Dataset):
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

        train_subjects = list(range(0, 18))
        test_subjects = list(range(18, 23))

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

        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
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

        self.seq_length = 50
        self.flattened = False

    def flatten_sequences(self):
        """Flattens (num_seq, seq_len, feat, win) → (num_seq*seq_len, feat, win)."""
        if not self.flattened:
            self.flattened = True

    def __len__(self):
        if self.flattened:
            return len(self.x_data)
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        if self.flattened:
            return self.x_data[idx].transpose(1,0), self.y_data[idx]

        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
        return x_seq, y_seq