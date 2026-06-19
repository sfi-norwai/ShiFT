from src.loader.dataloader import PAMAP2Dataset, ECGDataset2, ECGDataset3, ECGDatasetDeterministic
from src.loader.dataloader import WISDM2Dataset, SKODADataset, HARTHDataset, SleepmDataset

def preload_data(dataset):
    # Create the dataset
    if dataset == 'harth':
        train_ds = HARTHDataset(
                data_path=f'datasets/{dataset}',
                eval=False
            )
        
        valid_ds = HARTHDataset(
                data_path=f'datasets/{dataset}',
                eval=True
            )
        
    elif dataset == 'ecg2':
        train_ds = ECGDatasetDeterministic(
                data_path=f'datasets/{dataset}',
                eval=False
            )
        
        valid_ds = ECGDataset3(
                data_path=f'datasets/{dataset}',
                eval=True
            )
    
    elif dataset == 'pamap2':
        train_ds = PAMAP2Dataset(
                data_path=f'datasets/{dataset}',
                eval=False
            )
        
        valid_ds = PAMAP2Dataset(
                data_path=f'datasets/{dataset}',
                eval=True
            )

    elif dataset == 'skoda':
        train_ds = SKODADataset(
                data_path=f'datasets/{dataset}',
                eval=False
            )
        
        valid_ds = SKODADataset(
                data_path=f'datasets/{dataset}',
                eval=True
            )
            
    elif dataset == 'wisdm2':
        train_ds = WISDM2Dataset(
                data_path=f'datasets/{dataset}',
                eval=False
            )
        
        valid_ds = WISDM2Dataset(
                data_path=f'datasets/{dataset}',
                eval=True
            )
    
    elif dataset == 'sleepm':
        train_ds = SleepmDataset(
                data_path=f'datasets/{dataset}',
                eval=False
            )
        
        valid_ds = SleepmDataset(
                data_path=f'datasets/{dataset}',
                eval=True
            )
    else:
        raise ValueError(f"Unsupported DATASET: {dataset}")
    
    return train_ds, valid_ds
