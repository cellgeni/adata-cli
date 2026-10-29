"""Check that each direct dependency is installed at its floor.

The floors job in tests.yml resolves pyproject.toml with
`--resolution lowest-direct`. A floor the resolver stepped over without a
word (a release that was never published, or one that conflicts with
another requirement) would leave the job testing a version pyproject does
not name, so this compares what is installed with what is declared, and
imports each one.

    python .github/scripts/check_floors.py [EXTRA ...]

checks the dependencies and the named extras; with no extras, all of them.
"""

import importlib
import sys
import tomllib
from importlib.metadata import packages_distributions, version

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

with open("pyproject.toml", "rb") as f:
    project = tomllib.load(f)["project"]
extras = sys.argv[1:] or list(project["optional-dependencies"])
lines = list(project["dependencies"])
for extra in extras:
    lines += project["optional-dependencies"][extra]

floors: dict[str, Version] = {}
wrong = []
for line in lines:
    req = Requirement(line)
    found = [Version(s.version) for s in req.specifier if s.operator == ">="]
    if len(found) != 1:
        wrong.append(f"{line}: no single >= floor")
        continue
    name = canonicalize_name(req.name)
    # A package in two lists (pandas in pickle and dev) resolves to the
    # higher of its floors.
    floors[name] = max(found[0], floors.get(name, found[0]))

# Each distribution's own module, which is named after it or is the only one
# it installs (Pillow's PIL). pyarrow's list also holds directories of its
# source tree, which do not import.
modules: dict[str, list[str]] = {}
for module, dists in packages_distributions().items():
    for dist in dists:
        modules.setdefault(canonicalize_name(dist), []).append(module)

for name, floor in sorted(floors.items()):
    installed = version(name)
    if Version(installed) != floor:
        wrong.append(f"{name}: installed {installed}, floor {floor}")
        continue
    # The tests importorskip anndata, pandas, scipy and others, so a floor
    # that installs but does not import would skip them and pass.
    candidates = modules.get(name, [])
    own = name.replace("-", "_")
    module = own if own in candidates else (candidates + [None])[0]
    if module is None or (len(candidates) > 1 and module != own):
        wrong.append(f"{name}: cannot tell its module from {candidates}")
        continue
    try:
        importlib.import_module(module)
    except Exception as e:
        wrong.append(f"{name} {installed}: import {module} fails: {e!r}")
        continue
    print(f"{name} {installed}")

if wrong:
    sys.exit("not at the floor:\n  " + "\n  ".join(wrong))
