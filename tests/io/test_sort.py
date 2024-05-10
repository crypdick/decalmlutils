from collections import Counter

import hypothesis.strategies as strat
from hypothesis import given, settings

from decalmlutils.io.sort import natural_sort


@given(
    list_or_dict=strat.one_of(
        strat.lists(elements=strat.one_of(strat.integers(), strat.text())),
        strat.dictionaries(
            keys=strat.one_of(strat.integers(), strat.text()),
            values=strat.one_of(strat.integers(), strat.text()),
        ),
    )
)
@settings(max_examples=3)
def test_natural_sort(list_or_dict):
    sorted_list_or_dict = natural_sort(list_or_dict)

    # test properties we want to have
    assert Counter(sorted_list_or_dict) == Counter(list_or_dict)
    assert type(sorted_list_or_dict) == type(list_or_dict)
