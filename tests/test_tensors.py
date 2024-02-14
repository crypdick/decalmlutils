import hypothesis
import pytest
from hypothesis import example, given
from hypothesis import strategies as strat

from rd_utils.tensors import get_chunks


@given(
    num_items=strat.integers(min_value=1, max_value=1_000),
    chunk_size=strat.integers(1, 1_500),
)
@example(num_items=1, chunk_size=10)  # test this always
@example(num_items=1, chunk_size=1)  # test this always
@hypothesis.settings(max_examples=10)
def test_get_chunks(num_items, chunk_size):
    chunks = get_chunks(num_items, chunk_size)

    assert len(chunks) > 0
    last_chunk = chunks[-1]
    assert last_chunk[-1] == num_items
    if chunks:
        assert chunks[0][0] == 0

        # to test chunks are the right size, we need to ignore the final chunk
        chunks_no_last = chunks[:-1]
        if chunks_no_last:
            for chunk_min, chunk_max in chunks_no_last:
                assert chunk_max - chunk_min == chunk_size

        original = list(range(num_items))
        reconstructed = []
        for chunk_min, chunk_max in chunks:
            reconstructed.extend(original[chunk_min:chunk_max])
        assert (
            reconstructed == original
        ), f"Reconstructed list does not match original list of len {num_items}: {reconstructed} != {original} from chunks (size {chunk_size}): {chunks}"


def test_get_chunks_fixed():
    chunks = get_chunks(394283, 100_000)

    assert chunks == [(0, 100000), (100000, 200000), (200000, 300000), (300000, 394283)]

    with pytest.raises(AssertionError):
        get_chunks(-1, -1)
        get_chunks(100, -1)
        get_chunks(-1, 2)
