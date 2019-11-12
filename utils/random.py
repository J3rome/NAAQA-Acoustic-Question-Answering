import random

import torch
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # FIXME : Use manual_seed_all ? (For all gpus)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_random_state():
    states = {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state()
    }

    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state()

    return states


def set_random_state(states):
    random.setstate(states['py'])
    np.random.set_state(states['np'])
    torch.random.set_rng_state(states['torch'])

    if torch.cuda.is_available() and 'torch_cuda' in states:
        torch.cuda.set_rng_state(states['torch_cuda'])
