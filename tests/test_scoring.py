from __future__ import annotations

from analystagent.eval.scoring import compare_results, normalize_cell


def match(golden, candidate, ordered=False):
    return compare_results(golden, candidate, ordered)


def test_identical_results_match_strictly():
    result = match([("a", 1)], [("a", 1)])
    assert result.strict and result.lenient


def test_row_order_is_ignored_unless_ordered():
    golden = [("a", 1), ("b", 2)]
    shuffled = [("b", 2), ("a", 1)]
    assert match(golden, shuffled).strict
    assert not match(golden, shuffled, ordered=True).lenient


def test_ordered_results_match_in_order():
    rows = [("a", 1), ("b", 2)]
    assert match(rows, list(rows), ordered=True).strict


def test_floats_are_compared_at_two_decimals():
    assert match([(2328.6,)], [(2328.6000000000004,)]).strict
    assert match([(5.65,)], [(5.6516,)]).strict
    assert not match([(5.65,)], [(5.7,)]).lenient


def test_integer_valued_floats_equal_integers():
    assert match([(59,)], [(59.0,)]).strict


def test_numeric_strings_equal_numbers():
    assert match([("2021", 449.46)], [(2021, 449.46)]).strict


def test_leading_zero_strings_stay_strings():
    assert normalize_cell("007") == "007"
    assert normalize_cell("0") == 0
    assert normalize_cell(" text ") == "text"


def test_none_and_bool_normalisation():
    assert normalize_cell(None) is None
    assert normalize_cell(True) == 1


def test_extra_columns_pass_leniently_only():
    result = match([("USA", 13)], [("USA", 13, "extra")])
    assert not result.strict and result.lenient


def test_extra_column_in_front_passes_leniently():
    result = match([("USA", 13)], [(1, "USA", 13)])
    assert not result.strict and result.lenient


def test_permuted_columns_pass_leniently():
    result = match([("USA", 13)], [(13, "USA")])
    assert not result.strict and result.lenient


def test_missing_column_fails():
    assert not match([("USA", 13)], [("USA",)]).lenient


def test_wrong_value_fails():
    assert not match([("USA", 13)], [("USA", 14)]).lenient


def test_row_count_mismatch_fails():
    assert not match([(1,), (2,)], [(1,)]).lenient
    assert not match([(1,)], [(1,), (2,)]).lenient


def test_both_empty_match():
    assert match([], []).strict


def test_golden_rows_but_empty_candidate_fails():
    assert not match([(1,)], []).lenient


def test_lenient_projection_must_align_rows():
    golden = [("a", 1), ("b", 2)]
    swapped = [("a", 2), ("b", 1)]
    assert not match(golden, swapped).lenient


def test_lenient_projection_with_duplicate_columns():
    result = match([("a", "a")], [("a", "a", "a")])
    assert result.lenient


def test_type_confusion_between_text_and_number_is_not_a_match():
    assert not match([("abc",)], [(1,)]).lenient


def test_assignment_search_is_bounded():
    golden = [tuple([1] * 6)]
    candidate = [tuple([1] * 12)]
    assert match(golden, candidate).lenient
