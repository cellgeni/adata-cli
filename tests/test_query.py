"""Tests for the obs/var predicate language."""

from __future__ import annotations

import pytest

from adata.core.query import QueryError, compile_query, referenced_columns

COLUMNS = {
    "cluster": ["A", "B", "A", "C"],
    "n_counts": ["10", "200", "30", "4000"],
    "flag": ["True", "False", "True", "False"],
}


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("cluster == A", [True, False, True, False]),
        ("cluster != A", [False, True, False, True]),
        ("cluster in A,B", [True, True, True, False]),
        ("cluster not in A", [False, True, False, True]),
        ("n_counts > 100", [False, True, False, True]),
        ("n_counts >= 30", [False, True, True, True]),
        ("n_counts < 30", [True, False, False, False]),
        ("n_counts <= 30", [True, False, True, False]),
        ("flag == True", [True, False, True, False]),
        ("cluster == A and n_counts > 20", [False, False, True, False]),
        ("cluster == C or n_counts < 20", [True, False, False, True]),
        ("not cluster == A", [False, True, False, True]),
        ("(cluster == A or cluster == C) and n_counts > 20", [False, False, True, True]),
    ],
)
def test_predicates(expr, expected):
    assert compile_query(expr)(COLUMNS).tolist() == expected


def test_quoted_values_allow_spaces():
    columns = {"label": ["cell type A", "other"]}
    assert compile_query('label == "cell type A"')(columns).tolist() == [True, False]


def test_parenthesised_value_list():
    assert compile_query("cluster in (A, C)")(COLUMNS).tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_referenced_columns_ignores_values():
    assert referenced_columns("cluster == A and n_counts > 5") == [
        "cluster",
        "n_counts",
    ]
    assert referenced_columns("cluster in A,B") == ["cluster"]


def test_unknown_column_is_reported():
    with pytest.raises(QueryError, match="nope"):
        compile_query("nope == 1")(COLUMNS)


def test_ordering_operator_needs_a_number():
    with pytest.raises(QueryError, match="needs a number"):
        compile_query("cluster > abc")(COLUMNS)


def test_unparseable_numbers_do_not_match():
    """A non-numeric entry is simply excluded rather than raising."""
    columns = {"x": ["1", "not-a-number", "3"]}
    assert compile_query("x > 0")(columns).tolist() == [True, False, True]


def test_empty_query_rejected():
    with pytest.raises(QueryError):
        compile_query("   ")


def test_unbalanced_parentheses_rejected():
    with pytest.raises(QueryError):
        compile_query("(cluster == A")
