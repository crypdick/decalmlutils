import random
from typing import Optional

import numpy as np
import torch
from beartype import beartype


@beartype
def seed_everything(seed: Optional[int] = None):
    """
    Seeds all random number generators. Record the seed you used for reproducibility.

    Args:
        seed: an integer use for seeding the RNG
    """
    if not seed:
        seed = generate_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
    except AttributeError:  # CUDA not available
        pass


@beartype
def generate_seed() -> int:
    """
    Creates a random seed.

    Returns:
        seed: a random integer. This should be recorded for reproducibility.
    """
    MAX_ALLOWED_NUMPY_SEED = 2**32 - 1
    seed = np.random.randint(0, MAX_ALLOWED_NUMPY_SEED)
    return seed
