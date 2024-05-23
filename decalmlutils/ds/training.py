import random

import numpy as np
import torch
from beartype import beartype
from beartype.typing import Optional

from decalmlutils.misc import millify


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


def count_params(model):
    """
    Count the number of parameters in a model.
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"total params: {millify(total)}. Trainable params: {millify(total)}")
    return total, trainable


def compute_grad_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None and param.requires_grad
    ]

    grad_norm = torch.cat(grads).norm().item()

    return grad_norm
