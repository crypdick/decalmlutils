import random
from typing import Callable, Literal, Optional

import numpy as np
import torch
from beartype import beartype
from torchvision import transforms

TransformVersion = Literal["v1", "v2", "v3"]


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


@beartype
def _transforms_eval(
    mean: list[float],
    stddev: list[float],
    version: TransformVersion,
) -> Callable:
    """
    Minimal set of transforms for evaluation.
    """
    if version == "v1":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=stddev),
            ]
        )
    else:
        raise ValueError(f"Unknown version of transforms: {version}")
    return transform


@beartype
def create_transform(
    kind: Literal["train", "eval", "corrupt"],
    mean: Optional[list[float]],
    stddev: Optional[list[float]],
    version: TransformVersion = "v3",
    **kwargs,
) -> Callable:
    """
    A transforms factory; decides whether to create preprocessing, training, or evaluation transforms.

    Args:
        version: version of augmentations to use
        magnitude: training augmentation magnitude
        mean: training dataset mean
        stddev: training dataset standard devation
        kind: flag for kind of transform to generate

    Returns:
        PyTorch Transforms
    """

    if kind == "eval":
        transform = _transforms_eval(mean, stddev, version=version)
    else:
        raise ValueError(f"Unknown kind of transform: {kind}")

    return transform
