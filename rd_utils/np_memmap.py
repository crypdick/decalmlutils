import numpy as np
from pathlib import Path
from filelock import FileLock


def init_memmap(
    filepath: str,
    dataset_size: int,
    emd_size: int,
    dtype: str = "float32",
    mode="r",
    check_exists=False,
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array.

    Useful for working with huge arrays that do not fit in memory.

    Args:
        embs_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emd_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.
        mode (str): Mode to open the memory-mapped file in. r=ro existing, r+=rw existing, w+=rw and overwrite if exists.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    if mode in ["r", "r+"]:
        if check_exists:
            assert Path(filepath).exists(), f"File {filepath} does not exist"
    elif mode == "w+":
        print(f"Creating new memory-mapped file at {filepath}")
        assert (
            isinstance(dataset_size, int) and dataset_size > 0
        ), "Invalid dataset size"

    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"Initializing memmap at {filepath}...")
    with FileLock(str(filepath) + ".lock"):
        embs = np.memmap(
            filepath,
            dtype=dtype,
            mode=mode,
            shape=(dataset_size, emd_size),
        )
    return embs
