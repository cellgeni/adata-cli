"""Check that the documentation describes the CLI that actually exists.

Every `adata ...` invocation in the README and docs is extracted and its
options checked against the real `--help` for that command. This exists
because the command reference advertised `--zarr-format` on `split` before the option
was implemented: prose drifts from code silently, and nothing else in the
suite reads the docs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import pytest
from typer.main import get_command

from adata.cli import app

#: The real command tree. Options are read from here rather than from
#: rendered `--help` text: Rich wraps long option names at the terminal width,
#: so a help-scraping check passes on a wide terminal and fails in CI.
#: It is walked through typer's objects, never click's classes: typer 0.26
#: and later carry their own copy of click and no longer install click, so a
#: fresh install has none to import. A group is whatever has `commands`.
ROOT = get_command(app)


def _subcommands(command: Any) -> dict:
    """A group's subcommands by name; empty for a plain command."""
    return getattr(command, "commands", None) or {}


def _lookup(path: List[str]) -> Any:
    """Resolve a subcommand path, or raise KeyError naming what is missing."""
    command = ROOT
    for name in path:
        if not hasattr(command, "commands"):
            raise KeyError(f"{name!r}: {path} is not a group")
        found = _subcommands(command).get(name)
        if found is None:
            raise KeyError(f"{name!r} is not a command of {path}")
        command = found
    return command


def _options(command: Any) -> set:
    """Every option string the command accepts, long and short."""
    names = set()
    for param in command.params:
        names.update(param.opts)
        names.update(param.secondary_opts)
    return names

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "docs"
#: Every page of the site, at any depth: Guides and Reference are folders.
DOC_PAGES = sorted(SITE.rglob("*.md"))
DOCS = [REPO / "README.md"] + DOC_PAGES


def _label(path: Path) -> str:
    """A page's name in test ids and messages; several are `index.md`."""
    return str(path.relative_to(REPO))


def _front_matter(text: str) -> Tuple[dict, str]:
    """Jekyll front matter as a flat dict, and the page body after it.

    Only `key: value` lines are kept; a list such as `redirect_from` is read
    as its items under that key. Enough for the keys the sidebar uses.
    """
    if not text.startswith("---\n"):
        return {}, text
    head, _, body = text[4:].partition("\n---\n")
    found: dict = {}
    key = None
    for line in head.splitlines():
        item = re.match(r"^\s+-\s+(.*)$", line)
        if item and key:
            found.setdefault(key, []).append(item.group(1).strip())
            continue
        pair = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if pair:
            key, value = pair.group(1), pair.group(2).strip().strip('"')
            if value:
                found[key] = value
    return found, body

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
                found.append((line, _label(path)))
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
    "view", "info", "ls", "subset", "split", "concat", "convert", "recast",
    "create", "export", "import",
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
    # `adata --version`: an option of the app itself, checked against the root.
    assert path or tokens[1].startswith("-"), (
        f"{source}: could not identify a command in {line!r}"
    )

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
                    claims.append((option, command, _label(doc)))
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
# `benchmarks/` both went live as 404s on the 0.6.0 release page. Since the
# move to Just the Docs with `permalink: pretty`, `guides/subset.md` is served
# at `/guides/subset/`, one level below its file, so only a link to another
# page survives: jekyll-relative-links rewrites `page.md` to that page's URL
# and leaves everything else as written. Checked offline against the
# filesystem, so it costs nothing and needs no network.


def _without_code(text: str) -> str:
    """The page with fenced blocks blanked: `# comment` is not a heading."""
    return re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)


def _relative_links(text: str):
    """(link, target, fragment) for every relative markdown link."""
    for match in re.finditer(r"\]\(([^)\s]+)\)", _without_code(text)):
        link = match.group(1)
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, fragment = link.partition("#")
        yield link, target, fragment


def _heading_ids(page: Path) -> set:
    """The ids kramdown's GFM parser gives a page's headings.

    Lowercase, every character but a word character, `-`, space or tab
    dropped, spaces turned into `-`, and a repeat suffixed `-1`, `-2`: the
    same rule GitHub uses, so an anchor that works here works there too.
    """
    _, body = _front_matter(page.read_text())
    ids, seen = set(), {}
    for match in re.finditer(r"^#{1,6}\s+(.*?)\s*#*\s*$", _without_code(body), re.M):
        slug = re.sub(r"[^\w\- \t]", "", match.group(1).lower())
        slug = slug.replace(" ", "-").replace("\t", "-")
        seen[slug] = seen.get(slug, -1) + 1
        ids.add(slug + (f"-{seen[slug]}" if seen[slug] else ""))
    return ids


@pytest.mark.parametrize("page", DOC_PAGES, ids=_label)
def test_every_relative_link_resolves_inside_the_published_site(page):
    """A relative link must name a page the site serves, and an anchor on it.

    Splitting the command reference into a page per format moved dozens of
    anchors, and a link to a heading that is gone still loads the page, so
    nothing but a check like this one notices.
    """
    broken = []
    for link, target, fragment in _relative_links(page.read_text()):
        if not target:
            resolved = page
        else:
            resolved = (page.parent / target).resolve()
            if not resolved.is_relative_to(SITE.resolve()):
                broken.append(f"{link} -- leaves docs/, so Pages cannot serve it")
                continue
            if resolved.suffix != ".md":
                broken.append(
                    f"{link} -- only links to another page's .md are "
                    "rewritten; this one resolves against the page's pretty URL"
                )
                continue
            if not resolved.exists():
                broken.append(f"{link} -- no such page")
                continue
        if fragment and fragment not in _heading_ids(resolved):
            broken.append(f"{link} -- no heading with that anchor")

    assert not broken, (
        f"{_label(page)} has links that 404 on the published site:\n  "
        + "\n  ".join(broken)
        + "\nUse an absolute https://github.com/... URL for anything that is "
        "not a page of the site."
    )


# ---------------------------------------------------------------------------
# the sidebar
#
# Just the Docs builds its navigation from front matter alone: a page with no
# `title` is not in the sidebar, and one whose `parent` names no page's title
# is left out of it, with no warning from the build.

FRONT_MATTER = {page: _front_matter(page.read_text())[0] for page in DOC_PAGES}
TITLES = [fm.get("title") for fm in FRONT_MATTER.values()]


@pytest.mark.parametrize("page", DOC_PAGES, ids=_label)
def test_every_page_has_a_title_and_a_parent_that_exists(page):
    front = FRONT_MATTER[page]
    assert front.get("title"), f"{_label(page)} has no `title` front matter"
    for key in ("parent", "grand_parent"):
        if key in front:
            assert front[key] in TITLES, (
                f"{_label(page)}: {key} {front[key]!r} is no page's title"
            )
            assert TITLES.count(front[key]) == 1, (
                f"{_label(page)}: {key} {front[key]!r} is the title of "
                "several pages, so the sidebar cannot tell which"
            )
    if "grand_parent" in front:
        parent = next(
            fm for fm in FRONT_MATTER.values() if fm.get("title") == front["parent"]
        )
        assert parent.get("parent") == front["grand_parent"], (
            f"{_label(page)}: grand_parent {front['grand_parent']!r} is not "
            f"the parent of {front['parent']!r}"
        )


# ---------------------------------------------------------------------------
# every command is documented
#
# The tests above check one direction: that everything the docs claim really
# exists. They say nothing about the reverse, so `convert` shipped in 0.6.0
# with a section in the command reference that the table of contents never listed, and
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


COMMANDS_MD = _front_matter((SITE / "reference" / "commands.md").read_text())[1]


@pytest.mark.parametrize("name", _visible_commands())
def test_every_command_has_a_section_in_the_reference(name):
    assert f"## `{name}`" in COMMANDS_MD, (
        f"`adata {name}` has no `## \\`{name}\\`` section in docs/reference/commands.md"
    )


@pytest.mark.parametrize("name", _visible_commands())
def test_every_command_is_in_the_table_of_contents(name):
    contents = COMMANDS_MD.split("---", 1)[0]
    assert f"](#{name})" in contents, (
        f"`adata {name}` is missing from the table of contents in "
        "docs/reference/commands.md. The link text has no backticks: "
        "`- [name](#name)`."
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


# ---------------------------------------------------------------------------
# docstring examples
#
# The checks above cover the Markdown docs, but several `Examples:` blocks in
# cli.py drifted after the rename to `adata`: `export sparse`, `export dict`
# and `export image` still showed a `h5ad` prefix that never existed, and two
# of them took a positional output the commands reject. Nothing in the suite
# read the docstrings, so nothing caught it. The `--help` text is parsed the
# same way the docs are: example lines must use the `adata` name, resolve to
# a real command, and only use options the command accepts. Walking the real
# Click tree means a command added tomorrow is covered automatically.


def _help_examples() -> List[Tuple[str, str]]:
    """Every example-looking command line inside any command's `--help`."""
    found: List[Tuple[str, str]] = []

    def walk(command: Any, path: List[str]) -> None:
        for raw in (command.help or "").splitlines():
            line = raw.strip()
            # Same cleanup as the docs parser: comments, pipes, redirections.
            line = re.split(r"\s+#\s", line)[0].strip()
            line = re.split(r"\s*[|>]\s*", line)[0].strip()
            if line.startswith(("adata ", "h5ad ")):
                found.append((line, "cli.py:" + ("/".join(path) or "<app>")))
        for name, sub in _subcommands(command).items():
            if getattr(sub, "hidden", False):
                continue
            walk(sub, path + [name])

    walk(ROOT, [])
    return found


HELP_EXAMPLES = _help_examples()


def test_the_help_text_contains_examples_to_check():
    """A parsing regression here would silently make the test below vacuous."""
    assert len(HELP_EXAMPLES) > 10, HELP_EXAMPLES


def _problems_with_example(line: str) -> List[str]:
    """What is wrong with one `adata ...` example line from a docstring."""
    if line.startswith("h5ad "):
        return [
            "uses the pre-rename `h5ad` command name; the CLI is `adata` now"
        ]
    tokens = line.split()
    path = _command_path(tokens)
    if not path:
        return [f"could not identify a command in {line!r}"]
    try:
        command = _lookup(path)
    except KeyError as exc:
        return [str(exc)]
    accepted = _options(command)
    problems = []
    for token in tokens[len(path) + 1 :]:
        if token.startswith("--"):
            option = token.split("=")[0]
            if _PLACEHOLDER.search(option):
                continue
            if option not in accepted:
                problems.append(
                    f"uses {option}, which `adata {' '.join(path)}` "
                    "does not accept"
                )
        elif re.fullmatch(r"-[A-Za-z]", token):
            if token not in accepted:
                problems.append(
                    f"uses {token}, which `adata {' '.join(path)}` "
                    "does not accept"
                )
    return problems


@pytest.mark.parametrize(
    "line,source",
    HELP_EXAMPLES,
    ids=[f"{src}:{line[:60]}" for line, src in HELP_EXAMPLES],
)
def test_help_examples_name_real_commands_and_options(line, source):
    problems = _problems_with_example(line)
    assert not problems, f"{source}: `{line}` -- " + "; ".join(problems)
