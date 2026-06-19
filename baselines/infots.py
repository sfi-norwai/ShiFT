import torch
from tqdm import tqdm
from src.models.ts2vecencoder import *
import src.config, src.utils, src.models, src.hunt_data
from src.losses.contrastive import global_infoNCE, local_infoNCE
from src.models.infotsaugmentation import AutoAUG
from src.models.attention_model import *
from src.models.ts2vecencoder import *
from pytorch_lightning.loggers import WandbLogger
import wandb
#from models import TSEncoder
from src.src_utils.utils import cosine_warmup_scheduler
import time
from utils import name_with_datetime
import os

class InfoTS:
    '''The InfoTS model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a InfoTS model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        self._net = TSEncoder(input_dims=args['feature_dim'], output_dims=args['out_features']).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.n_iters = 0
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the InfoTS model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        
        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size= self.args['batch_size'],
                shuffle = True,
                num_workers=self.config.NUM_WORKERS,
                drop_last = True,
            )
        
        # Wandb setup
        if self.config.WANDB:    
            proj_name = 'Dynamic_CL' + ds_name + str(self.config.SEED)
            run_name = 'InfoTS'

            wandb_logger = WandbLogger(project=proj_name)
            
            # Initialize Wandb
            wandb.init(project=proj_name, name=run_name)
            wandb.watch(self.net, log='all', log_freq=100)

            # Update Wandb config
        
            wandb.config.update(self.args)
            wandb.config.update({
                'Algorithm': f'{run_name}',
                'Dataset': f'{ds_name}',
                'Train_DS_size': len(train_dataset),
                'Batch_Size': self.args["batch_size"],
                'Epochs': self.args["epochs"],
                'Patience': self.config.PATIENCE,
                'Seed': self.config.SEED

            })
            wandb.run.name = run_name
            wandb.run.save()
        
        # Define loss function and optimizer
        self.args['lr'] = float(self.args['lr'])
        self.args['weight_decay'] = float(self.args['weight_decay'])

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args['lr'], betas=(0.9, 0.99), weight_decay=self.args['weight_decay'])
    
        
        max_train_length = None
        aug = AutoAUG(device=self.device)
        t0 = 2.0
        t1 = 0.1

        n_iters = self.args['iterations']
        pbar = tqdm(total=n_iters, desc="Training")
        epoch = 0
        num_training_steps = n_iters
        num_warmup_steps = int(0.1 * n_iters)

        scheduler = cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)

        if self.args['save_model']:
            run_dir = f'results/{ds_name}/seed_{self.config.SEED}/{name_with_datetime(self.__class__.__name__)}'
            os.makedirs(run_dir, exist_ok=True)
            start_time = time.time()

        while True:

            # Training phase
            self.net.train()  # Set the model to training mode
            train_running_loss = 0.0
            n_epoch_iters = 1

            for x, _ in train_loader:

                interrupted = False
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
               
                if max_train_length is not None and x.size(1) > max_train_length:
                    window_offset = np.random.randint(x.size(1) - max_train_length + 1)
                    x = x[:, window_offset : window_offset + max_train_length]
                x = x.to(self.device) 

                
                if n_epoch_iters==-1:
                    t =1.0
                else:
                    t = float(t0 * np.power(t1 / t0, (n_epoch_iters+1) / n_epoch_iters))

                a1,a2 = aug((x,t))

                out1 = self.net(a1.to(self.device))
                out2 = self.net(a2.to(self.device))

                # Calculate the loss
                loss = global_infoNCE(
                    out1, out2, self.device) + local_infoNCE(out1, out2, self.device, k=8)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
    
                # Update training statistics
                n_epoch_iters += 1
                self.n_iters += 1
                pbar.update(1)

                train_running_loss += loss.item()

            scheduler.step()
            if interrupted:
                break
            train_running_loss /= n_epoch_iters
    
            if verbose:
                print(f"Epoch {epoch}, Train Loss: {train_running_loss:.4f}")

            # Log training loss to Wandb
            if self.config.WANDB:
                wandb.log({'Train Loss': train_running_loss, 'Epoch': epoch})

        # Save model
        if self.args['save_model']:
            model_path = os.path.join(run_dir, f'model.pt')
            torch.save(self.net.state_dict(), model_path)

            total_time = time.time() - start_time

            # Save training time
            time_file = os.path.join(run_dir, 'time.txt')
            with open(time_file, 'w') as f:
                f.write(str(total_time))

        try:   
            return train_running_loss
        except:
            return 0
    
    def encode(self, x, mask=None):

        self.net.eval()
        out = self.net(x.to(self.device, non_blocking=True), mask)

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
