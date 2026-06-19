import torch
import numpy as np
import argparse
import os
import time
import datetime

from baselines.rand_init import Rand_Init
from baselines.simclr import SimCLR
from baselines.shift import ShiFT
from baselines.ts2vec import TS2Vec
from baselines.infots import InfoTS
from baselines.simmtm import SimMTM
from utils import name_with_datetime, pkl_save
import wandb
import tasks
import argparse
import random
import src.data
from src.loader.dataloader import PAMAP2Dataset, ECGDataset2, ECGDataset3
from src.loader.dataloader import WISDM2Dataset, SKODADataset, HARTHDataset, SleepmDataset
from special_tasks.evaluate_model import eval_supervised
from src.loader.dataloader import stratified_fixed_count
import hashlib
import json
from mdl import InceptionTime


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start Vanilla CL training.')
    parser.add_argument('model', help='The model name')
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/sleepconfig.yml')
    parser.add_argument('-s', '--seed_value', required=False, type=int,
                        help='seed value.', default=42)
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        help='batch value.', default=8) 
    parser.add_argument('-v', '--verbose_bool', required=False, type=bool,
                        help='verbose bool.', default=False)
    parser.add_argument('-g', '--gpu', required=False, type=int,
                        help='int.', default=0)
    parser.add_argument('-th', '--max_threads', required=False, type=int,
                        help='number of threads.', default=8)
    parser.add_argument('-iter', '--iterations', required=False, type=int,
                        help='number of iterations.', default=None)
    parser.add_argument('--evaluate', required=False, type=str,
                        help='Task to evaluate on.', default=None)
    
    parser.add_argument('-sp', '--semi_percentage', required=False, type=int,
                        help='percentage of training data.', default=0.01)
    
    parser.add_argument('-trans', '--transfer_data', required=False, type=str,
                        help='The transfer dataset name', default='ecg')
    
   
    
    
    
    pargs = parser.parse_args()
    config_path = pargs.params_path
    # Read config
    config = src.config.Config(config_path)
    config.SEED = pargs.seed_value
    ds_path = pargs.dataset
    verbose = pargs.verbose_bool
    gpu_val = pargs.gpu
    max_threads = pargs.max_threads
    
    # Log in to Wandb
    if config.WANDB:
        wandb.login(key=config.WANDB_KEY)
    
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
            
            seed = config.SEED
            if pargs.iterations is not None:
                 args['iterations'] = pargs.iterations
            args['save_model'] = True
            args['lr'] = 1e-3 

            #args['batch_size'] = 32
            #args['lr'] = 1e-3 * (args['batch_size'] / 128)
            #args['iterations'] = 1500 * (128 / args['batch_size'])

           # args['feature_dim'] = 3

            device = torch.device(f"cuda:{pargs.gpu}" if torch.cuda.is_available() else "cpu")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            # Set all seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            
            # Create the dataset
            if config.DATASET == 'HARTH':
                train_ds = HARTHDataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = HARTHDataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )
                
            elif config.DATASET == 'ECG2':
                train_ds = ECGDataset2(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = ECGDataset3(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )
            
            elif config.DATASET == 'PAMAP2':
                train_ds = PAMAP2Dataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = PAMAP2Dataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )

            elif config.DATASET == 'SKODA':
                train_ds = SKODADataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = SKODADataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )
                   
            elif config.DATASET == 'WISDM2':
                train_ds = WISDM2Dataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = WISDM2Dataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )
            
            elif config.DATASET == 'SLEEPM':
                train_ds = SleepmDataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = SleepmDataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )

            else:
                raise ValueError(f"Unsupported DATASET: {config.DATASET}")

            
            
            t = time.time()

            
            if pargs.model == 'ShiFT':
                model = ShiFT(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'SimCLR':
                model = SimCLR(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'TS2Vec':
                model = TS2Vec(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'InfoTS':
                model = InfoTS(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'SimMTM':
                model = SimMTM(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'Rand_Init':
                model = Rand_Init(
                    args,
                    config,
                    device=device
                )

            else:
                raise ValueError(f"Unsupported BASELINE: {pargs.model}")

            #feature_idx = [3, 4, 5]  # select 3 out of 6
            #train_ds = FeatureSubsetDataset(train_ds, feature_idx)
            #valid_ds = FeatureSubsetDataset(valid_ds, feature_idx)

            
            loss_log = model.fit(train_ds,ds_path,verbose=pargs.verbose_bool)
            
            # model.save(f'{run_dir}/model.pkl')

            # t = time.time() - t
            # print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

            # Select fixed samples per class
            # print("Original size:", len(train_ds))
            # train_ds = stratified_fixed_count(train_ds, n_per_class=50)
            # print("Subset size:", len(train_ds))

            # trained_model = train_supervised_pretrain(model, train_ds, ds_args['num_labels'], args, config, device=device)

            if pargs.evaluate:
                if pargs.evaluate == 'supervised':
                    if pargs.model in ['SimMTM','SimCLR','ShiFT','Rand_Init']:
                        eval_res = tasks.supervised_evaluation_ponly_block(model, train_ds, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], config)
                    elif pargs.model == 'SupervisedE2E':
                        eval_res = eval_supervised(model, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], config)
                    else:
                        eval_res = tasks.supervised_evaluation_ponly(model, train_ds, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], config)
                
                elif pargs.evaluate == 'semi_supervised':
                    eval_res = tasks.semi_supervised_evaluation(model, train_ds, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], pargs.semi_percentage/100, config)
                elif pargs.evaluate == 'clustering':
                    eval_res = tasks.clustering_evaluation(model, valid_ds, config)
                else:
                    assert False

                # Save evaluation results
                results = {
                   "args": args,
                    "metrics": {
                    "acc": eval_res[0],
                    "f1": eval_res[1],
                                }
                }
                
                def hash_args(args_dict):
                    hp_string = json.dumps(args_dict, sort_keys=True)
                    return hashlib.md5(hp_string.encode()).hexdigest()[:8]

                exp_id = hash_args(args)

                save_dir = f"hyperparameters/{pargs.model}/{pargs.dataset}/{exp_id}"
                os.makedirs(save_dir, exist_ok=True)

                filename = f"seed_{pargs.seed_value}.json"
                with open(os.path.join(save_dir, filename), "w") as f:
                    json.dump(results, f, indent=4)
                print('Evaluation result:', eval_res)

                
    print("Finished.")
