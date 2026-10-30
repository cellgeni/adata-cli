"""Tests for the obs/var predicate language."""

from __future__ import annotations

import numpy as np
import pytest

from adata.core.query import Column, QueryError, compile_query, referenced_columns

pd = pytest.importorskip("pandas")

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


# ---------------------------------------------------------------------------
# value lists (F012)


CELL_TYPES = [
    "CD4-positive, alpha-beta T cell",
    "B cell",
    "CD8-positive, alpha-beta T cell",
    "alpha-beta T cell",
]


@pytest.mark.parametrize(
    "expr",
    [
        'ct in ("CD4-positive, alpha-beta T cell", "B cell")',
        "ct in ('CD4-positive, alpha-beta T cell','B cell')",
        'ct in ("CD4-positive, alpha-beta T cell" "B cell")',
        'ct in "CD4-positive, alpha-beta T cell", "B cell"',
    ],
)
def test_a_quoted_label_with_a_comma_is_one_label(expr):
    """F012: `in` split every value at its commas, quoted or not, so
    `"CD4-positive, alpha-beta T cell"` became `CD4-positive` and
    `alpha-beta T cell`: the CD4 cells were missed and the plain
    `alpha-beta T cell` rows matched instead, with exit 0."""
    wanted = pd.Series(CELL_TYPES).isin(["CD4-positive, alpha-beta T cell", "B cell"])
    assert compile_query(expr)({"ct": CELL_TYPES}).tolist() == wanted.tolist()
    negated = expr.replace(" in ", " not in ", 1)
    assert compile_query(negated)({"ct": CELL_TYPES}).tolist() == (~wanted).tolist()


@pytest.mark.parametrize(
    "expr", ["cluster in A,C", "cluster in A, C", "cluster in (A,C)", "cluster in (A C)"]
)
def test_bare_words_in_a_list_are_still_separated_by_commas(expr):
    wanted = pd.Series(COLUMNS["cluster"]).isin(["A", "C"])
    assert compile_query(expr)(COLUMNS).tolist() == wanted.tolist()


def test_a_quoted_empty_label_can_be_listed():
    """Quoted values are taken literally, so `""` is a label in a list, as
    it is after `==`; it used to be dropped as an empty item."""
    labels = ["", "a", "b"]
    assert compile_query('x in ("", a)')({"x": labels}).tolist() == [True, True, False]


def test_an_unquoted_comma_after_equality_asks_for_quotes():
    with pytest.raises(QueryError, match="quote"):
        compile_query("ct == CD4-positive, alpha-beta")


# ---------------------------------------------------------------------------
# operators without spaces (F119)

TYPED = {
    "n_counts": np.array([10, 200, 30, 4000], dtype=np.int64),
    "cluster": np.array(["A", "B", "A", "C"]),
}
FRAME = pd.DataFrame(TYPED)


@pytest.mark.parametrize(
    "expr",
    [
        "n_counts>100",
        "n_counts>=30",
        "n_counts<30",
        "n_counts<=-1",
        "cluster==A",
        "cluster!=A",
        "cluster==A and n_counts>20",
        "(cluster==A or cluster==C)and n_counts>20",
        "not(n_counts>=200)",
    ],
)
def test_operators_need_no_spaces_around_them(expr):
    """F119: a bare word ran up to the next space, so `n_counts>1000` was
    one word and the query failed with 'Unexpected end of query.', which
    gave no hint that spaces were needed. pandas reads these the same way."""
    wanted = FRAME.eval(expr.replace("==A", "=='A'").replace("==C", "=='C'")
                        .replace("!=A", "!='A'"))
    assert compile_query(expr)(TYPED).tolist() == wanted.tolist()


@pytest.mark.parametrize("expr,hint", [("cluster = A", "=="), ("n_counts =< 3", "=="),
                                       ("n_counts ! 3", "!=")])
def test_a_lone_equals_or_bang_says_which_operators_there_are(expr, hint):
    with pytest.raises(QueryError, match=f"Unknown operator.*{hint}"):
        compile_query(expr)


def test_an_unclosed_quote_is_reported_as_one():
    with pytest.raises(QueryError, match="quote"):
        compile_query('cluster == "A')


# ---------------------------------------------------------------------------
# numbers and booleans compare by value (F120)

VALUES = pd.DataFrame(
    {
        "total_counts": np.array([0, 1000, 2.5, 0], dtype=np.float32),
        "pct_mt": np.array([0.1, 0.3, 0.1, 0.7], dtype=np.float32),
        "n_genes": np.array([0, 3, 7, 0], dtype=np.int32),
        "is_doublet": np.array([True, False, False, True]),
        "score": np.array([1.0, np.nan, 2.0, np.nan]),
        "small": np.array([1, 2, 3, 4], dtype=np.int8),
    }
)
COLUMN_CHUNK = {name: Column(VALUES[name].to_numpy()) for name in VALUES}


@pytest.mark.parametrize(
    "expr,pandas_expr",
    [
        ("total_counts != 0", None),
        ("total_counts == 0", None),
        ("total_counts == 1000", None),
        ("total_counts == 1e3", "total_counts == 1000"),
        ("total_counts > 0", None),
        ("pct_mt == 0.1", None),
        ("pct_mt <= 0.1", None),
        ("pct_mt > 0.1", None),
        ("n_genes != 0", None),
        ("n_genes == 3.0", None),
        ("n_genes < 2.5", None),
        ("n_genes in (0, 7)", "n_genes in [0, 7]"),
        ("n_genes not in 0,7", "n_genes not in [0, 7]"),
        ("is_doublet == true", "is_doublet == True"),
        ("is_doublet == TRUE", "is_doublet == True"),
        ("is_doublet != true", "is_doublet != True"),
        ("is_doublet == False", None),
        ("is_doublet == 1", "is_doublet == True"),
        ("is_doublet in (false)", "is_doublet == False"),
        ("score != 1", None),
        ("score >= 1", None),
        ("small == 1000", None),
        ("small < 1000", None),
        ("small > -1000", None),
    ],
)
def test_numbers_and_booleans_compare_by_value_as_pandas_does(expr, pandas_expr):
    """F120: equality compared each value's repr with the text typed, so
    `total_counts != 0` kept the zeros of a float32 column (0.0 is not
    '0'), `is_doublet == true` matched nothing, and float32 values of 0.1
    missed `pct_mt <= 0.1` because they were widened to 0.10000000149011612
    first. Values now compare at the column's dtype, as pandas compares them."""
    wanted = VALUES.eval(pandas_expr or expr)
    assert compile_query(expr)(COLUMN_CHUNK).tolist() == wanted.tolist()


def test_nan_selects_the_missing_values_of_a_number_column():
    """A missing number (NaN, or masked in a nullable column) equals no
    value, so `!=` keeps it as pandas does, and `== nan` is how to ask for it."""
    masked = Column(np.array([1, 0, 3, 1], dtype=np.int64),
                    missing=np.array([False, True, False, False]))
    chunk = dict(COLUMN_CHUNK, masked=masked)
    expected = pd.Series(pd.array([1, None, 3, 1], dtype="Int64"))
    query = compile_query
    assert query("score == nan")(chunk).tolist() == VALUES["score"].isna().tolist()
    assert query("score != NaN")(chunk).tolist() == VALUES["score"].notna().tolist()
    assert query("masked == nan")(chunk).tolist() == expected.isna().tolist()
    assert query("masked != 1")(chunk).tolist() == (expected != 1).fillna(True).tolist()
    assert query("masked == 1")(chunk).tolist() == (expected == 1).fillna(False).tolist()
    assert query("masked > 0")(chunk).tolist() == (expected > 0).fillna(False).tolist()
    assert query("masked in (3, nan)")(chunk).tolist() == (
        expected.isin([3]) | expected.isna()
    ).tolist()


@pytest.mark.parametrize(
    "expr,message",
    [
        ("total_counts == abc", "holds numbers"),
        ("n_genes in (1, two)", "holds numbers"),
        ("is_doublet == yes", "true or false"),
        ("total_counts == 1_000", "holds numbers"),
    ],
)
def test_a_value_that_is_not_of_the_columns_type_is_refused(expr, message):
    """`!=` with a typo used to keep every row; it is refused instead."""
    with pytest.raises(QueryError, match=message):
        compile_query(expr)(COLUMN_CHUNK)
