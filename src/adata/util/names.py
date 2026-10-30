"""Names: turning labels into filenames, and duplicates into unique names."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set


def sanitize_label(label: Optional[str]) -> str:
    """Turn a column value into something safe to use as a filename.

    A missing value (None), and a label with nothing filename-safe in it,
    become `NA`.
    """
    text = (label or "").strip()
    if not text:
        return "NA"
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "", text)
    return text or "NA"


def unique_names(labels: Sequence[Optional[str]]) -> Dict[Optional[str], str]:
    """Map each label to a distinct filename stem, in the order given.

    Sanitising can map two different labels onto the same stem ("a/b" and
    "a b"), and a suffixed stem can be a later label's own ("CD4+", "CD4",
    "CD4_1"), so each stem is claimed against every stem given out before
    it. They are compared without case: `T` and `t` are one file on macOS
    and Windows. Either collision put two outputs on one path.
    """
    taken: Set[str] = set()
    mapping: Dict[Optional[str], str] = {}
    for label in labels:
        stem = candidate = sanitize_label(label)
        count = 0
        while candidate.casefold() in taken:
            count += 1
            candidate = f"{stem}_{count}"
        taken.add(candidate.casefold())
        mapping[label] = candidate
    return mapping


def storable(name: str) -> str:
    """`name` as an HDF5 or Zarr member can hold it: `/` separates their
    paths, and "", "." and ".." name the group itself or its parent."""
    return f"_{name}" if name in ("", ".", "..") else name.replace("/", "_")


def claim(name: str, taken: Set[str]) -> str:
    """`name`, or the first of `name_1`, `name_2`, ... not in `taken`.

    The name returned is added to `taken`, so claiming several in turn
    gives distinct names.
    """
    candidate, count = name, 0
    while candidate in taken:
        count += 1
        candidate = f"{name}_{count}"
    taken.add(candidate)
    return candidate


def count_duplicates(names: Sequence[str]) -> int:
    """How many names repeat one that came before them."""
    return len(names) - len(set(names))


def make_index_unique(names: Sequence[str], join: str = "-") -> List[str]:
    """anndata's `make_index_unique`, without pandas.

    The first occurrence of a name is kept; each later one gets `-1`, `-2`,
    and so on, skipping any suffixed name that already exists. Matching
    anndata exactly matters: names made unique here and names made unique
    by `var_names_make_unique` after reading must be the same names.
    """
    values = list(names)
    taken = set(values)
    if len(taken) == len(values):
        return values

    seen: set = set()
    counter: Counter = Counter()
    for i, value in enumerate(values):
        if value not in seen:
            seen.add(value)
            continue
        while True:
            counter[value] += 1
            candidate = f"{value}{join}{counter[value]}"
            if candidate not in taken:
                taken.add(candidate)
                values[i] = candidate
                break
    return values


_DIGITS = re.compile(r"(\d+)")


def natural_key(text: str) -> tuple:
    """Sort key putting "A2" before "A10", as natsort's default does."""
    parts = _DIGITS.split(text)
    return tuple(int(part) if i % 2 else part for i, part in enumerate(parts))


def natural_sorted(values: Iterable[str]) -> List[str]:
    """`values` in natsort's default order, which is how anndata orders the
    categories it makes from strings (`strings_to_categoricals`). Equal keys
    ("01" and "1") keep their plain sorted order, as natsort's stable sort
    of pandas' sorted categories does."""
    return sorted(sorted(values), key=natural_key)
