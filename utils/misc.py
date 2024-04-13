import random

import numpy as np
import torch
import yaml
from easydict import EasyDict


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)