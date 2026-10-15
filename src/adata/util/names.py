"""Names: turning labels into filenames, and duplicates into unique names."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence


def sanitize_label(label: str) -> str:
    """Turn a column value into something safe to use as a filename."""
    text = (label or "").strip()
    if not text:
        return "NA"
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "", text)
    return text or "NA"


def unique_names(labels: List[str]) -> Dict[str, str]:
    """Map each label to a distinct filename stem.

    Sanitising can map two different labels onto the same stem ("a/b" and
    "a b"), so collisions get a numeric suffix rather than silently
    overwriting one another.
    """
    used: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    for label in labels:
        base = sanitize_label(label)
        count = used.get(base, 0)
        used[base] = count + 1
        mapping[label] = base if count == 0 else f"{base}_{count}"
    return mapping


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
