"""
Functions that are used on array like variables/objects.
"""

import numpy as np
import torch
from beartype import beartype
from beartype.typing import Iterable, Tuple, Union
from PIL import Image as img
from PIL.Image import Image
from torch import Tensor

IMG_OPAQUE_THRESH = 0.9  # ignore images with less than this fraction opaque pixels


@beartype
def get_chunks(num_items: int, chunk_size: int = 100_000) -> list[Tuple[int, int]]:
    """
    Produces chunk indices for metaflow for_each fan-outs.

    Example usage:
    Chunking a list with 394283 items:

    get_chunks(394283) --> [(0, 100000), (100000, 200000), (200000, 300000), (300000, 394283)]

    NOTE: Python slicing logic does not include the final index. That is why the chunks seemingly overlap in the bounds.
    If convert the outputs of this function for naming chunks, add 1 to each slice index min, e.g.
    chunk_names = [(chunk_min + 1, chunk_max) for chunk_min, chunk_max in chunk_indices]
    """
    assert num_items > 0
    assert chunk_size > 0

    chunks = []
    for i in range(0, num_items, chunk_size):
        chunks.append((i, i + chunk_size))

    # correct final chunk's range
    last_chunk_min, _last_chunk_max = chunks[-1]
    chunks[-1] = (last_chunk_min, num_items)

    return chunks


@beartype
def arr_to_tensor(arr: Iterable) -> Tensor:
    """
    Convert iterable into a Torch Tensor.
    """
    if isinstance(arr, Image):
        raise ValueError("arr_to_tensor: use convert_image_type() for PIL Images")

    if not isinstance(arr, Tensor):
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)  # share memory; avoid copy
        else:
            arr = Tensor(arr)
    return arr


@beartype
def array_to_pil(arr: Union[np.ndarray, Image]) -> Image:
    if isinstance(arr, Image):
        return arr
    else:
        return img.fromarray(arr)


@beartype
def arr_to_npy(arr: Union[Iterable, Image]) -> np.ndarray:
    """
    Convert array into numpy array.
    """
    if isinstance(arr, np.ndarray):
        pass
    elif isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    elif isinstance(arr, Image):
        # gotcha: need to use np.array() instead of np.asarray() to avoid memory sharing
        # otherwise, we won't be able to set flags
        # this sucks, because it means we have to copy the array
        arr = np.array(arr, copy=True)
    elif not isinstance(arr, np.ndarray):
        arr = np.array(arr, copy=True)
    else:
        raise ValueError(f"arr_to_npy: unrecognized type {type(arr)}")

    try:
        arr.setflags(write=1)  # prevent not-writeable error when converting to tensor
    except ValueError:
        # 2023-5-16 I do not know how np arrays make it to this stage without owning their memory
        # but it happens, and this is a hacky fix
        arr = np.array(arr, copy=True)

    return arr


@beartype
def threshold_confidences(arr: Iterable, thresh: float) -> np.ndarray:
    """
    Given an array of floats, convert to binary values using a threshold.
    """
    assert 0 <= thresh <= 1
    arr = arr_to_npy(arr)

    arr = arr > thresh  # sklearn expects hard labels
    arr = arr.astype(int)  # sklearn metrics expect int dtype
    return arr


@beartype
def np_array_elements(array: np.ndarray) -> int:
    """
    Counts the number of elements in a numpy array.

    https://stackoverflow.com/a/56715195/4212158
    """
    return array.ndim and array.size


@beartype
def is_transparent_arr(img: np.ndarray) -> bool:
    """
    Given a RGBA ndarry in channels_last memory format, return True if the image is transparent, False otherwise.
    """
    assert img.dtype == np.uint8, f"Image array is not uint8: dtype {img.dtype}"
    assert img.shape[-1] == 4, f"Image array is not RGBA: shape {img.shape}"

    alpha_chan = img[:, :, -1]
    num_opaque = np.count_nonzero(alpha_chan == 255)  # 255 is opaque
    frac_opaque = num_opaque / alpha_chan.size

    assert 0 <= frac_opaque <= 1

    is_transparent = frac_opaque < IMG_OPAQUE_THRESH

    return is_transparent
