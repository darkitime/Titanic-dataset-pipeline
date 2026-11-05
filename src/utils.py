import random
import os
import torch
import numpy as np
from configs.config_CL import config_CL

def set_seed(seed: int = config_CL.general.seed):
    '''
    Setting seed for determination
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False