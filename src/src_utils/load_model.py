
from baselines.shift import ShiFT

from baselines.simclr import SimCLR

from baselines.ts2vec import TS2Vec

from baselines.infots import InfoTS

from baselines.simmtm import SimMTM
from baselines.rand_init import Rand_Init

def initialize_model(method, args, config, ds_args, device):

    

    if method == 'SimCLR':
        model = SimCLR(
            args,
            config,
            device=device
        )
    
    elif method == 'ShiFT':
        model = ShiFT(
            args,
            config,
            device=device
        )

    elif method == 'TS2Vec':
        model = TS2Vec(
            args,
            config,
            device=device
        )

    elif method == 'InfoTS':
        model = InfoTS(
            args,
            config,
            device=device
        )

    elif method == 'SimMTM':
        model = SimMTM(
            args,
            config,
            device=device
        )

    elif method == 'Rand_Init':
        model = Rand_Init(
            args,
            config,
            device=device
        )
    
    else:
        raise ValueError(f"Unsupported BASELINE: {method}")

    return model
