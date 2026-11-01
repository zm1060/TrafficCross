"""Training seeds, independent of the fixed dataset split seed."""

import random

import numpy as np
import torch


def configure_training_seed(config, seed_override=None):
    """Seed Python, NumPy, and PyTorch and record the resolved training seed."""
    seed = seed_override if seed_override is not None else config['training'].get('seed', 17)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Training seed must be an integer between 0 and 2**32 - 1')
    config['training']['seed'] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
