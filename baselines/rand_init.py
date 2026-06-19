from torch.nn import GRU, Linear, CrossEntropyLoss
import wandb
import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from tqdm import tqdm
# from src.models.attention_model import *
from src.models.inceptiontime_pool import *
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from src.src_utils.utils import cosine_warmup_scheduler
import time
from utils import name_with_datetime
import os
from src.models.resnet1D import *

class Rand_Init:

    '''The Rand_Init model'''
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a Rand_Init model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        # self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)
        self.net = InceptionTime(n_in_channels=args['feature_dim'], out_channels=args['out_features']).to(self.device)
        #self.net = ResNet1D(n_in_channels=args['feature_dim'], out_channels=args['out_features']).to(self.device)

        self.n_iters = 0
        

       
    
    def fit(self, train_dataset, ds_name, verbose=False):

        epoch = 1

        run_dir = f'results/{ds_name}/seed_{self.config.SEED}/{name_with_datetime(self.__class__.__name__)}'
        os.makedirs(run_dir, exist_ok=True)
        start_time = time.time()

       
        # Save model
        model_path = os.path.join(run_dir, f'model.pt')
        torch.save(self.net.state_dict(), model_path)

        total_time = time.time() - start_time

        # Save training time
        time_file = os.path.join(run_dir, 'time.txt')
        with open(time_file, 'w') as f:
            f.write(str(total_time))
            
        try:   
            return 0
        except:
            return 0
    
    def encode(self, x):
        self.net.eval()
        out = self.net(x.to(self.device))

        return out


    def save(self, fn):
        ''' Save the model to a file.
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
