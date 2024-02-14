"""
Functions that are used on array like variables/objects.
"""

from typing import List, Tuple


from beartype import beartype


@beartype
def get_chunks(num_items: int, chunk_size: int = 100_000) -> List[Tuple[int, int]]:
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
