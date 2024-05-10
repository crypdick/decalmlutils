from beartype import beartype
from beartype.typing import Callable, Literal, Optional
from torchvision import transforms

TransformVersion = Literal["v1", "v2", "v3"]
TransformType = Literal["train", "eval", "corrupt"]


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
    kind: TransformType,
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
