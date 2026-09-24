"""Check that the documentation describes the CLI that actually exists.

Every `adata ...` invocation in the README and docs is extracted and its
options checked against the real `--help` for that command. This exists
because COMMANDS.md advertised `--zarr-format` on `split` before the option
was implemented: prose drifts from code silently, and nothing else in the
suite reads the docs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Tuple

import click
import pytest
from typer.main import get_command

from adata.cli import app

#: The real Click command tree. Options are read from here rather than from
#: rendered `--help` text: Rich wraps long option names at the terminal width,
#: so a help-scraping check passes on a wide terminal and fails in CI.
ROOT = get_command(app)


def _lookup(path: List[str]) -> click.Command:
    """Resolve a subcommand path, or raise KeyError naming what is missing."""
    command: click.Command = ROOT
    for name in path:
        if not isinstance(command, click.Group):
            raise KeyError(f"{name!r}: {path} is not a group")
        found = command.get_command(click.Context(command), name)
        if found is None:
            raise KeyError(f"{name!r} is not a command of {path}")
        command = found
    return command


def _options(command: click.Command) -> set:
    """Every option string the command accepts, long and short."""
    names = set()
    for param in command.params:
        names.update(param.opts)
        names.update(param.secondary_opts)
    return names

REPO = Path(__file__).resolve().parent.parent
DOCS = [REPO / "README.md"] + sorted((REPO / "docs").glob("*.md"))

#: Placeholders that stand in for a real path in the docs.
_PLACEHOLDER = re.compile(r"^[<{\[]|[>}\]]$")


def _fenced_blocks(text: str) -> Iterator[str]:
    """Yield the contents of ```bash / ```shell fences."""
    for match in re.finditer(r"```(?:bash|shell|console|sh)\n(.*?)```", text, re.S):
        yield match.group(1)


def _invocations(path: Path) -> List[Tuple[str, str]]:
    """Every `adata ...` command line in a document, with its source line."""
    found: List[Tuple[str, str]] = []
    for block in _fenced_blocks(path.read_text()):
        for raw in block.splitlines():
            line = raw.strip().rstrip("\\").strip()
            # Strip a trailing comment and any shell redirection or pipe.
            line = re.split(r"\s+#\s", line)[0].strip()
            line = re.split(r"\s*[|>]\s*", line)[0].strip()
            if line.startswith("adata ") and not line.startswith("adata-cli"):
                found.append((line, path.name))
    return found


ALL_INVOCATIONS = [inv for doc in DOCS for inv in _invocations(doc)]


def test_the_docs_contain_examples_to_check():
    """A parsing regression here would silently make every case below vacuous."""
    assert len(ALL_INVOCATIONS) > 20, ALL_INVOCATIONS


def _command_path(tokens: List[str]) -> List[str]:
    """The subcommand path, e.g. ['export', 'dataframe']."""
    path: List[str] = []
    for token in tokens[1:]:
        if token.startswith("-"):
            break
        # A value rather than a subcommand name.
        if path and path[0] in {"export", "import"} and len(path) == 2:
            break
        if not path or path[0] in {"export", "import"}:
            if _looks_like_a_subcommand(token, path):
                path.append(token)
                continue
        break
    return path


_GROUPS = {"export", "import"}
_TOP_LEVEL = {
    "view", "ls", "subset", "split", "concat", "convert", "create",
    "export", "import",
}
_SUBCOMMANDS = {
    "export": {"dataframe", "array", "sparse", "dict", "image"},
    "import": {"dataframe", "array", "sparse", "dict", "image"},
}


def _looks_like_a_subcommand(token: str, path: List[str]) -> bool:
    if not path:
        return token in _TOP_LEVEL
    if path[0] in _GROUPS and len(path) == 1:
        return token in _SUBCOMMANDS[path[0]]
    return False


@pytest.mark.parametrize(
    "line,source",
    ALL_INVOCATIONS,
    ids=[f"{src}:{line[:60]}" for line, src in ALL_INVOCATIONS],
)
def test_documented_invocations_use_real_commands_and_options(line, source):
    tokens = line.split()
    path = _command_path(tokens)
    assert path, f"{source}: could not identify a command in {line!r}"

    try:
        command = _lookup(path)
    except KeyError as exc:
        pytest.fail(f"{source}: `{line}` -- {exc}")

    accepted = _options(command)
    for token in tokens[len(path) + 1 :]:
        if not token.startswith("--"):
            continue
        option = token.split("=")[0]
        if _PLACEHOLDER.search(option):
            continue
        assert option in accepted, (
            f"{source}: `{line}` uses {option}, which `adata "
            f"{' '.join(path)}` does not accept. Accepted: "
            f"{', '.join(sorted(o for o in accepted if o.startswith('--')))}"
        )


def test_short_options_in_the_docs_exist_too():
    """Short flags are easy to mistype and just as easy to check."""
    problems = []
    for line, source in ALL_INVOCATIONS:
        tokens = line.split()
        path = _command_path(tokens)
        if not path:
            continue
        accepted = _options(_lookup(path))
        for token in tokens[len(path) + 1 :]:
            if not re.fullmatch(r"-[A-Za-z]", token):
                continue
            if token not in accepted:
                problems.append(f"{source}: `{line}` uses {token}")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# prose claims
#
# The fenced-invocation checks above only see commands someone wrote out in
# full. The `--zarr-format` drift lived in a sentence instead -- "`--zarr-format
# 2|3` on `create`, `subset`, `split` and `concat`" -- so sentences that name an
# option and the commands offering it are checked too.

_CLAIM = re.compile(
    r"`(--[a-z][a-z0-9-]*)[^`]*`[^.]*?\bon\b((?:[^.]|\n)*?)(?:overrides|applies|works)",
    re.I,
)
_COMMAND_IN_PROSE = re.compile(r"`([a-z][a-z0-9-]*)`")


def _prose_claims() -> List[Tuple[str, str, str]]:
    """(option, command, source) for each sentence claiming support."""
    claims: List[Tuple[str, str, str]] = []
    for doc in DOCS:
        for match in _CLAIM.finditer(doc.read_text()):
            option = match.group(1)
            for command in _COMMAND_IN_PROSE.findall(match.group(2)):
                if command in _TOP_LEVEL:
                    claims.append((option, command, doc.name))
    return claims


PROSE_CLAIMS = _prose_claims()


def test_prose_claims_were_found():
    assert PROSE_CLAIMS, "the claim parser matched nothing; it has rotted"


@pytest.mark.parametrize(
    "option,command,source",
    PROSE_CLAIMS,
    ids=[f"{src}:{cmd}{opt}" for opt, cmd, src in PROSE_CLAIMS],
)
def test_options_claimed_in_prose_exist(option, command, source):
    try:
        resolved = _lookup([command])
    except KeyError as exc:
        pytest.fail(f"{source}: {exc}")
    assert option in _options(resolved), (
        f"{source}: prose says {command} supports {option}, "
        f"but the command does not accept it"
    )


# ---------------------------------------------------------------------------
# links that have to work on the published site
#
# GitHub Pages serves `docs/` as the site root, so a relative link out of it
# cannot resolve: `../.github/workflows/benchmark.yml` and a bare
# `benchmarks/` both went live as 404s on the 0.6.0 release page. Checked
# offline against the filesystem, so it costs nothing and needs no network.


def _relative_links(text: str):
    """(link, target) for every relative markdown link, fragments stripped."""
    for match in re.finditer(r"\]\(([^)\s]+)\)", text):
        link = match.group(1)
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = link.split("#", 1)[0]
        if target:
            yield link, target


DOC_PAGES = sorted((REPO / "docs").glob("*.md"))


@pytest.mark.parametrize("page", DOC_PAGES, ids=lambda p: p.name)
def test_every_relative_link_resolves_inside_the_published_site(page):
    """A relative link must point at something Pages actually serves.

    Jekyll rewrites `TESTING.md` to `/TESTING.html`, so a `.md` target is
    fine; anything reached with `../`, or a bare directory with no index,
    is not, and has to be an absolute URL to GitHub instead.
    """
    broken = []
    for link, target in _relative_links(page.read_text()):
        resolved = (page.parent / target).resolve()
        try:
            inside = resolved.is_relative_to((REPO / "docs").resolve())
        except AttributeError:  # pragma: no cover - Python < 3.9
            inside = str(resolved).startswith(str((REPO / "docs").resolve()))

        if not inside:
            broken.append(f"{link} -- leaves docs/, so Pages cannot serve it")
        elif not resolved.exists():
            broken.append(f"{link} -- no such file")
        elif resolved.is_dir() and not (resolved / "index.md").exists():
            broken.append(f"{link} -- a directory with no index page")

    assert not broken, (
        f"{page.name} has links that 404 on the published site:\n  "
        + "\n  ".join(broken)
        + "\nUse an absolute https://github.com/... URL for anything outside "
        "docs/."
    )


# ---------------------------------------------------------------------------
# every command is documented
#
# The tests above check one direction: that everything the docs claim really
# exists. They say nothing about the reverse, so `convert` shipped in 0.6.0
# with a section in COMMANDS.md that the table of contents never listed, and
# an app help string that named neither it nor `create`. Both were invisible
# to a suite that only walks from the docs to the code.
#
# Derived from the Click tree rather than a hardcoded list, so a command
# added tomorrow is covered without anyone remembering to add it here.


def _visible_commands():
    import typer

    command = typer.main.get_command(app)
    return sorted(
        name for name, sub in command.commands.items() if not sub.hidden
    )


COMMANDS_MD = (REPO / "docs" / "COMMANDS.md").read_text()


@pytest.mark.parametrize("name", _visible_commands())
def test_every_command_has_a_section_in_the_reference(name):
    assert f"## `{name}`" in COMMANDS_MD, (
        f"`adata {name}` has no `## \\`{name}\\`` section in docs/COMMANDS.md"
    )


@pytest.mark.parametrize("name", _visible_commands())
def test_every_command_is_in_the_table_of_contents(name):
    contents = COMMANDS_MD.split("---", 1)[0]
    assert f"](#{name})" in contents, (
        f"`adata {name}` is missing from the table of contents in "
        "docs/COMMANDS.md. The link text has no backticks: `- [name](#name)`."
    )


@pytest.mark.parametrize("name", _visible_commands())
def test_every_command_is_named_in_the_app_help(name):
    """`adata --help` opens with a list; a command absent from it is hidden
    in plain sight, since that line is the first thing anyone reads."""
    import typer

    help_text = typer.main.get_command(app).help or ""
    assert name in help_text, (
        f"`{name}` is missing from the app help string in cli.py: {help_text!r}"
    )
