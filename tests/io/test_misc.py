from datetime import datetime

import hypothesis.strategies as strat
import pytest
from hypothesis import given, settings

from mltoolkit.io.misc import (
    all_elements_same,
    flatten_list,
    format_date,
    is_dict_equal,
    kwargs_to_filename,
    millify,
)
from mltoolkit.training import generate_seed


@given(
    number=strat.one_of(
        strat.integers(min_value=0),
        strat.floats(min_value=0, allow_infinity=False, allow_nan=False),
    ),
    num_decimals=strat.integers(min_value=0, max_value=2),
)
@settings(max_examples=10)
def test_millify(number, num_decimals):
    millified = millify(number, num_decimals)

    if 0 < abs(number) < 1000:
        assert " " not in millified
        assert millified[-1].isnumeric()
        assert millified == f"{number:.{num_decimals}f}"
    elif abs(number) >= 1000:
        round_thousands = str(number).endswith("000")
        if num_decimals == 0 and not round_thousands:
            assert "." not in millified
        elif num_decimals > 0 and not round_thousands:
            assert "." in millified

        if 1000 <= number < 1_000_000:
            assert millified.endswith(" K")
        elif 1_000_000 <= number < 1_000_000_000:
            assert millified.endswith(" M")
        elif 1_000_000_000 <= number < 1_000_000_000_000:
            assert millified.endswith(" B")
        elif number >= 1_000_000_000_000:
            assert millified.endswith(" T")

    if number < 0:
        assert millified.startswith("-")

    # check first number after decimal is correct
    assert millify(9900, 1) == "9.9 K"
    assert millify(-9900, 1) == "-9.9 K"
    assert millify(9300, 0) == "9 K"
    assert millify(-9300, 0) == "-9 K"
    assert millify(9900, 0) == "10 K"
    assert millify(-9900, 0) == "-10 K"


@given(
    list_of_lists=strat.recursive(
        strat.lists(
            elements=strat.lists(
                strat.one_of(
                    strat.booleans(), strat.text(), strat.integers(), strat.floats()
                )
            ),
            max_size=2,
        ),
        extend=strat.lists,
        max_leaves=2,
    )
)
@settings(max_examples=10)
def test_list_flatten_contains_no_lists(list_of_lists):
    flattened_list = flatten_list(list_of_lists)

    assert all([not isinstance(element, list) for element in flattened_list])


def test_seed_generator_is_random():
    seeds = []
    for i in range(100):
        seeds.append(generate_seed())

    # there is an astronomically small chance this will fail
    assert len(set(seeds)) == len(seeds)


def test_is_dict_equal():
    equal_dict1 = {"a": 1, "b": 2, "c": 3}
    equal_dict2 = {"a": 1, "b": 2, "c": 3}
    same_key_diff_values_dict = {"a": 333333, "b": 2, "c": 3}
    some_same_keys_dict = {"axxxxxx": 1, "b": 2, "c": 3}
    all_diff_keys_dict = {"d": 4, "e": 5, "f": 6}

    assert is_dict_equal(equal_dict1, equal_dict2)
    assert not is_dict_equal(equal_dict1, same_key_diff_values_dict)
    assert not is_dict_equal(equal_dict1, some_same_keys_dict)
    assert not is_dict_equal(equal_dict1, all_diff_keys_dict)
    assert not is_dict_equal(same_key_diff_values_dict, equal_dict2)
    assert not is_dict_equal(same_key_diff_values_dict, some_same_keys_dict)
    assert not is_dict_equal(same_key_diff_values_dict, all_diff_keys_dict)
    assert not is_dict_equal(all_diff_keys_dict, equal_dict2)
    assert not is_dict_equal(all_diff_keys_dict, some_same_keys_dict)
    assert not is_dict_equal(some_same_keys_dict, equal_dict2)


def test_all_elements_same():
    assert all_elements_same([1] * 10)
    assert all_elements_same(["asd"] * 10)
    assert all_elements_same([1.1] * 10)
    assert not all_elements_same([1] * 9 + [2])
    assert not all_elements_same(["asd"] * 9 + ["x"])
    assert not all_elements_same([1.1] * 9 + [2.1])


def test_format_date():
    assert format_date("2021-01-12T10:30:00Z") == datetime(2021, 1, 12, 10, 30)
    assert format_date("2021-01-12T10:30:00.01") == datetime(
        2021, 1, 12, 10, 30, 0, 10000
    )
    assert format_date("2021-01-12T10:30:00.01Z") == datetime(
        2021, 1, 12, 10, 30, 0, 10000
    )


def test_kwargs_to_filename():
    # test deterministic order
    result = kwargs_to_filename(b=2, a=1, date=None)
    assert result == "a=1+b=2"

    # test date
    result = kwargs_to_filename(b=1, c=2, date="date")
    assert "b=1+c=2" in result
    # can't hard-code a date in a test, so just check that Y-m-d is in the result
    assert result[:4].isdigit()
    assert result[4] == "-"
    assert result[5:7].isdigit()
    assert result[7] == "-"
    assert result[8:10].isdigit()

    # test datetime
    result = kwargs_to_filename(b=1, c=2, date="datetime")
    assert "b=1+c=2" in result
    # can't hard-code a date in a test, so just check that "%Y-%m-%dT%H-%M-%S-%f"
    # is in the result
    assert result[:4].isdigit()
    assert result[4] == "-"
    assert result[5:7].isdigit()
    assert result[7] == "-"
    assert result[8:10].isdigit()
    assert result[10] == "T"
    assert result[11:13].isdigit()
    assert result[13] == "-"
    assert result[14:16].isdigit()
    assert result[16] == "-"
    assert result[17:19].isdigit()
    assert result[19] == "-"
    assert result[20:22].isdigit()

    # test that key_only works
    result = kwargs_to_filename(b=1, c=2, key_only=["g", "h"])
    assert result == "g+h+b=1+c=2"

    # test that "+ + + + +" is removed
    result = kwargs_to_filename(b=1, c=2, key_only=["+ + + + +"])
    assert result == "b=1+c=2"

    # test that illegal chars are replaced with a space
    blacklisted = ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]
    result = kwargs_to_filename(a=1, key_only=blacklisted)
    assert result == "a=1"

    # test that we add ext
    result = kwargs_to_filename(a=1, ext="txt")
    assert result == "a=1.txt"

    with pytest.raises(AssertionError):
        kwargs_to_filename()  # empty filename
        kwargs_to_filename(ext="txt")  # empty filename
