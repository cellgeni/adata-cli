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

import pytest
from typer.testing import CliRunner

from adata.cli import app

runner = CliRunner()

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
    "view", "ls", "subset", "split", "concat", "create", "export", "import",
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

    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, f"{source}: `{' '.join(path)}` is not a command"

    help_text = result.stdout
    for token in tokens[len(path) + 1 :]:
        if not token.startswith("--"):
            continue
        option = token.split("=")[0]
        if _PLACEHOLDER.search(option):
            continue
        assert option in help_text, (
            f"{source}: `{line}` uses {option}, which `adata "
            f"{' '.join(path)} --help` does not offer"
        )


def test_short_options_in_the_docs_exist_too():
    """Short flags are easy to mistype and just as easy to check."""
    problems = []
    for line, source in ALL_INVOCATIONS:
        tokens = line.split()
        path = _command_path(tokens)
        if not path:
            continue
        help_text = runner.invoke(app, [*path, "--help"]).stdout
        for token in tokens[len(path) + 1 :]:
            if not re.fullmatch(r"-[A-Za-z]", token):
                continue
            if token not in help_text:
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
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, f"{source}: `{command}` is not a command"
    assert option in result.stdout, (
        f"{source}: prose says {command} supports {option}, "
        f"but `adata {command} --help` does not offer it"
    )
