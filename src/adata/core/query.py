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
    comparison := IDENT OP VALUE
    OP      := == | != | < | <= | > | >= | in | not in

Values are bare words, quoted strings, or comma-separated lists for `in`.
Comparison against a column is done on its string form for equality and
membership, and numerically for the ordering operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

_TOKEN = re.compile(
    r"""
    \s*(?:
        (?P<lparen>\()
      | (?P<rparen>\))
      | (?P<op>==|!=|<=|>=|<|>)
      | (?P<word>"[^"]*"|'[^']*'|[^\s()]+)
    )
    """,
    re.VERBOSE,
)

_KEYWORDS = {"and", "or", "not", "in"}


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
            if expr[pos:].strip() == "":
                break
            raise QueryError(f"Cannot parse query at {expr[pos:]!r}")
        pos = match.end()
        for kind in ("lparen", "rparen", "op", "word"):
            text = match.group(kind)
            if text is None:
                continue
            if kind == "word" and text.lower() in _KEYWORDS:
                tokens.append(Token(text.lower(), text.lower()))
            else:
                tokens.append(Token(kind, text))
            break
    return tokens


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


#: A predicate maps a column reader to a boolean mask over the chunk.
Predicate = Callable[[Dict[str, List[str]]], np.ndarray]


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
        if self.peek() is not None:
            raise QueryError(f"Unexpected {self.peek().text!r} in query.")
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

        value_token = self.take()
        if value_token.kind != "word":
            raise QueryError(f"Expected a value, got {value_token.text!r}.")
        return _comparison(column, op_token.text, _unquote(value_token.text))

    def _take_list(self) -> List[str]:
        token = self.take()
        if token.kind == "lparen":
            items: List[str] = []
            while True:
                nxt = self.take()
                if nxt.kind == "rparen":
                    break
                items.extend(
                    v.strip() for v in _unquote(nxt.text).split(",") if v.strip()
                )
            return items
        if token.kind != "word":
            raise QueryError(f"Expected a value list, got {token.text!r}.")
        return [v.strip() for v in _unquote(token.text).split(",") if v.strip()]


def _combine(left: Predicate, right: Predicate, op: Any) -> Predicate:
    return lambda cols: op(left(cols), right(cols))


def _column(cols: Dict[str, List[str]], name: str) -> np.ndarray:
    if name not in cols:
        raise QueryError(
            f"Column {name!r} not found. Available: {', '.join(sorted(cols))}"
        )
    return np.asarray(cols[name], dtype=str)


def _as_float(values: np.ndarray, column: str) -> np.ndarray:
    """Parse a string column as floats, treating unparseable entries as NaN."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, v in enumerate(values):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _comparison(column: str, op: str, value: str) -> Predicate:
    def run(cols: Dict[str, List[str]]) -> np.ndarray:
        values = _column(cols, column)
        if op == "==":
            return values == value
        if op == "!=":
            return values != value

        try:
            threshold = float(value)
        except ValueError as exc:
            raise QueryError(
                f"Operator {op!r} needs a number, but got {value!r}."
            ) from exc

        numeric = _as_float(values, column)
        with np.errstate(invalid="ignore"):
            if op == "<":
                return numeric < threshold
            if op == "<=":
                return numeric <= threshold
            if op == ">":
                return numeric > threshold
            return numeric >= threshold

    return run


def _membership(column: str, options: Sequence[str], negate: bool) -> Predicate:
    wanted = set(options)

    def run(cols: Dict[str, List[str]]) -> np.ndarray:
        values = _column(cols, column)
        mask = np.isin(values, list(wanted))
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
