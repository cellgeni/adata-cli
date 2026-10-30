"""A small predicate language for selecting obs/var rows.

Deliberately not SQL. The point is to express the common filters -- one or two
comparisons on annotation columns -- without pulling a query engine into a tool
whose selling point is streaming with light dependencies. Anything more
involved is better served by exporting to CSV and using duckdb.

Grammar::

    expr    := or_expr
    or_expr := and_expr ( "or" and_expr )*
    and_expr:= term ( "and" term )*
    term    := "not" term | "(" expr ")" | comparison
    comparison := IDENT OP VALUE | IDENT ["not"] "in" LIST
    OP      := == | != | < | <= | > | >=
    LIST    := "(" VALUE ( [","] VALUE )* ")" | VALUE ( "," VALUE )*

A value is a bare word or a quoted string. A bare word ends at a space, a
parenthesis, a comma or an operator character, so `n_counts>1000` is three
tokens; a value holding any of those is quoted, and a quoted value is taken
literally, commas and all.

A column is compared at the type it is stored as. Numbers and booleans
compare by value, at the column's dtype, as pandas compares them: a float32
column meets `0.1` as float32 0.1. A missing number equals no value, so
`!=` keeps it, and `nan` names it. Text compares by its string, and a
missing text entry reads as "". The ordering operators parse text entries
as numbers, and an entry that will not parse does not match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

_TOKEN = re.compile(
    r"""
    \s*(?:
        (?P<lparen>\()
      | (?P<rparen>\))
      | (?P<comma>,)
      | (?P<op>==|!=|<=|>=|<|>)
      | (?P<word>"[^"]*"|'[^']*'|[^\s()=!<>,"'][^\s()=!<>,]*)
    )
    """,
    re.VERBOSE,
)

_KEYWORDS = {"and", "or", "not", "in"}

_OPERATORS = "==, !=, <, <=, > or >="


class QueryError(ValueError):
    """Raised when a query cannot be parsed or refers to a missing column."""


@dataclass
class Token:
    kind: str
    text: str


def tokenize(expr: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(expr):
        match = _TOKEN.match(expr, pos)
        if match is None:
            rest = expr[pos:].strip()
            if rest == "":
                break
            if rest[0] in "=!":
                raise QueryError(f"Unknown operator at {rest!r}: use {_OPERATORS}.")
            if rest[0] in "\"'":
                raise QueryError(f"Unclosed quote at {rest!r}.")
            raise QueryError(f"Cannot parse query at {rest!r}")
        pos = match.end()
        for kind in ("lparen", "rparen", "comma", "op", "word"):
            text = match.group(kind)
            if text is None:
                continue
            if kind == "word" and text.lower() in _KEYWORDS:
                # The text as typed: where a value belongs, a keyword is
                # one (`state in (OR, WA)`).
                tokens.append(Token(text.lower(), text))
            else:
                tokens.append(Token(kind, text))
            break
    return tokens


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


@dataclass
class Column:
    """One chunk of a column, at the type it is stored as.

    `values` holds numbers, booleans or text (numpy `U`). `missing` is True
    where an entry is missing: masked in a nullable column, NaN, or a
    category code of -1; a missing text entry holds "". `text`, where set,
    is how the entries read as text, when that is not `values` rendered by
    `str` (a categorical's numeric categories, as stored).
    """

    values: np.ndarray
    missing: Optional[np.ndarray] = None
    text: Optional[np.ndarray] = None

    def labels(self) -> np.ndarray:
        """The entries as text, as `split` names its outputs."""
        if self.text is not None:
            return self.text
        if self.values.dtype.kind == "U":
            return self.values
        return np.asarray([str(v) for v in self.values.tolist()], dtype=str)

    def is_missing(self) -> np.ndarray:
        missing = np.zeros(len(self.values), dtype=bool)
        if self.missing is not None:
            missing |= self.missing
        if self.values.dtype.kind == "f":
            missing |= np.isnan(self.values)
        return missing


#: A predicate maps a chunk of columns to a boolean mask over its rows. A
#: column is a `Column`, or a plain sequence (read as one with no gaps).
Predicate = Callable[[Dict[str, Any]], np.ndarray]


class _Parser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = list(tokens)
        self.pos = 0

    def peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> Token:
        token = self.peek()
        if token is None:
            raise QueryError("Unexpected end of query.")
        self.pos += 1
        return token

    def parse(self) -> Predicate:
        node = self.parse_or()
        token = self.peek()
        if token is not None and token.kind == "comma":
            raise QueryError(
                "Unexpected ',' in query: a value holding a comma needs quotes, "
                "as in label == \"a, b\"."
            )
        if token is not None:
            raise QueryError(f"Unexpected {token.text!r} in query.")
        return node

    def parse_or(self) -> Predicate:
        left = self.parse_and()
        while self.peek() is not None and self.peek().kind == "or":
            self.take()
            right = self.parse_and()
            left = _combine(left, right, np.logical_or)
        return left

    def parse_and(self) -> Predicate:
        left = self.parse_term()
        while self.peek() is not None and self.peek().kind == "and":
            self.take()
            right = self.parse_term()
            left = _combine(left, right, np.logical_and)
        return left

    def parse_term(self) -> Predicate:
        token = self.peek()
        if token is None:
            raise QueryError("Unexpected end of query.")

        if token.kind == "not":
            self.take()
            inner = self.parse_term()
            return lambda cols: np.logical_not(inner(cols))

        if token.kind == "lparen":
            self.take()
            inner = self.parse_or()
            closing = self.take()
            if closing.kind != "rparen":
                raise QueryError("Unbalanced parentheses in query.")
            return inner

        return self.parse_comparison()

    def parse_comparison(self) -> Predicate:
        name_token = self.take()
        if name_token.kind != "word":
            raise QueryError(f"Expected a column name, got {name_token.text!r}.")
        column = _unquote(name_token.text)

        op_token = self.take()
        if op_token.kind == "not":
            following = self.take()
            if following.kind != "in":
                raise QueryError("Expected 'in' after 'not'.")
            return _membership(column, self._take_list(), negate=True)
        if op_token.kind == "in":
            return _membership(column, self._take_list(), negate=False)
        if op_token.kind != "op":
            raise QueryError(
                f"Expected a comparison operator after {column!r}, "
                f"got {op_token.text!r}."
            )

        return _comparison(column, op_token.text, self._take_value())

    def _take_value(self) -> str:
        token = self.take()
        return self._value(token, "Expected a value")

    @staticmethod
    def _value(token: Token, expected: str) -> str:
        if token.kind == "word":
            return _unquote(token.text)
        if token.kind in _KEYWORDS:
            return token.text
        raise QueryError(f"{expected}, got {token.text!r}.")

    def _take_list(self) -> List[str]:
        """The values after `in`: in parentheses, separated by commas or
        spaces, or bare and separated by commas. Each is one token, so a
        quoted value keeps its commas."""
        token = self.peek()
        if token is not None and token.kind == "lparen":
            self.take()
            items: List[str] = []
            while True:
                nxt = self.take()
                if nxt.kind == "rparen":
                    return items
                if nxt.kind == "comma":
                    continue
                items.append(self._value(nxt, "Expected a value in the list"))
        items = [self._take_value()]
        while self.peek() is not None and self.peek().kind == "comma":
            self.take()
            items.append(self._take_value())
        return items


def _combine(left: Predicate, right: Predicate, op: Any) -> Predicate:
    return lambda cols: op(left(cols), right(cols))


def _column(cols: Dict[str, Any], name: str) -> Column:
    if name not in cols:
        raise QueryError(
            f"Column {name!r} not found. Available: {', '.join(sorted(cols))}"
        )
    column = cols[name]
    if isinstance(column, Column):
        return column
    values = np.asarray(column)
    return Column(values.astype(str) if values.dtype.kind in "OST" else values)


def _as_float(values: np.ndarray, column: str) -> np.ndarray:
    """Parse a string column as floats, treating unparseable entries as NaN."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, v in enumerate(values):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            continue
    return out


_INTEGER = re.compile(r"[+-]?\d+")
_NUMBER = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:inf|infinity|nan)",
    re.IGNORECASE,
)


def _number(text: str) -> Optional[Union[int, float]]:
    """`text` as a number, or None. An integer stays an int, so it compares
    exactly with an int64 column; Python-only spellings such as `1_000` are
    not numbers here."""
    text = text.strip()
    if _INTEGER.fullmatch(text):
        return int(text)
    if _NUMBER.fullmatch(text):
        return float(text)
    return None


def _boolean(text: str) -> Optional[bool]:
    folded = text.strip().lower()
    if folded in ("true", "1"):
        return True
    if folded in ("false", "0"):
        return False
    return None


def _is_nan(text: str) -> bool:
    return text.strip().lower() in ("nan", "+nan", "-nan")


def _kind(column: Column) -> str:
    kind = column.values.dtype.kind
    if kind == "b":
        return "boolean"
    if kind in "iuf":
        return "number"
    return "text"


def _equal(column: Column, name: str, value: str) -> np.ndarray:
    """Where `column` equals `value`: at its stored type, and never where an
    entry is missing, unless `value` is `nan`."""
    values = column.values
    kind = _kind(column)
    if kind == "text":
        return values == value
    if _is_nan(value):
        return column.is_missing()
    if kind == "boolean":
        target = _boolean(value)
        if target is None:
            raise QueryError(
                f"Column {name!r} holds true/false values; compare it with "
                f"true or false, not {value!r}."
            )
    else:
        target = _number(value)
        if target is None:
            raise QueryError(
                f"Column {name!r} holds numbers, so it compares with a number, "
                f"not {value!r}."
            )
    # A Python number meets a numpy array at the array's dtype, so a float32
    # column compares with float32(0.1), not with the float64 0.1.
    with np.errstate(over="ignore", invalid="ignore"):
        result = values == target
    return result & ~column.is_missing()


_ORDERING = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


def _comparison(column: str, op: str, value: str) -> Predicate:
    def run(cols: Dict[str, Any]) -> np.ndarray:
        values = _column(cols, column)
        if op == "==":
            return _equal(values, column, value)
        if op == "!=":
            return np.logical_not(_equal(values, column, value))

        threshold = _number(value)
        if threshold is None:
            raise QueryError(f"Operator {op!r} needs a number, but got {value!r}.")
        kind = _kind(values)
        if kind == "text":
            numeric = _as_float(values.values, column)
        elif kind == "boolean":
            numeric = values.values.astype(np.int8)
        else:
            numeric = values.values
        with np.errstate(over="ignore", invalid="ignore"):
            result = _ORDERING[op](numeric, threshold)
        return result & ~values.is_missing() if kind != "text" else result

    return run


def _membership(column: str, options: Sequence[str], negate: bool) -> Predicate:
    wanted = list(dict.fromkeys(options))

    def run(cols: Dict[str, Any]) -> np.ndarray:
        values = _column(cols, column)
        if _kind(values) == "text":
            mask = np.isin(values.values, wanted)
        else:
            mask = np.zeros(len(values.values), dtype=bool)
            for option in wanted:
                mask |= _equal(values, column, option)
        return np.logical_not(mask) if negate else mask

    return run


def compile_query(expr: str) -> Predicate:
    """Compile a query string into a predicate over a chunk of columns."""
    tokens = tokenize(expr)
    if not tokens:
        raise QueryError("Query is empty.")
    return _Parser(tokens).parse()


def referenced_columns(expr: str) -> List[str]:
    """Names that look like column references, so only those need reading."""
    tokens = tokenize(expr)
    names: List[str] = []
    for i, token in enumerate(tokens):
        if token.kind != "word":
            continue
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if following is not None and following.kind in ("op", "in", "not"):
            name = _unquote(token.text)
            if name not in names:
                names.append(name)
    return names
