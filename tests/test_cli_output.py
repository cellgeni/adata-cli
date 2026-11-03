"""What the CLI prints: names as the store spells them, errors where a pipeline
looks for them, help text that keeps its brackets.

Rich reads `[word]` in anything it prints as a style tag. Store member names,
labels and paths can hold square brackets (`pct_counts[mt]` is scanpy's own
QC column), so every such string has to be escaped before it reaches Rich, or
it is silently rewritten, or crashes the command on a closing tag.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from adata import cli
from adata.cli import app

runner = CliRunner()

#: Names scanpy and users really write, and ones Rich would read as tags.
OBS_COLUMNS = ["pct_counts[mt]", "qc[bold]", "[red]flag"]
OBSM_KEY = "X_[red]umap"
LABELS = ["s[1]", "[red]b", "x[/y]"]


def _bracketed_store(path: Path) -> Path:
    """An h5ad whose obs columns, obsm key and labels hold square brackets."""
    obs = pd.DataFrame(
        {
            OBS_COLUMNS[0]: np.arange(6.0),
            OBS_COLUMNS[1]: list("ababab"),
            OBS_COLUMNS[2]: [True, False] * 3,
            "batch[1]": pd.Categorical([LABELS[0]] * 2 + [LABELS[1]] * 2 + [LABELS[2]] * 2),
        },
        index=[f"c{i}" for i in range(6)],
    )
    adata = ad.AnnData(np.ones((6, 3), dtype="float32"), obs=obs)
    adata.obsm[OBSM_KEY] = np.zeros((6, 2))
    adata.write_h5ad(path)
    # An attribute value holding a closing tag: `view ENTRY` prints attributes.
    with h5py.File(path, "a") as f:
        f[f"obsm/{OBSM_KEY}"].attrs["note"] = "[/oops]"
    return path


@pytest.fixture
def bracketed(tmp_path) -> Path:
    return _bracketed_store(tmp_path / "names.h5ad")


def test_view_and_ls_print_member_names_with_square_brackets_as_they_are(bracketed):
    """F042: Rich took `[mt]` in `pct_counts[mt]` for a style tag, so `view`
    listed `pct_counts` and `ls` showed `X_umap` for `X_[red]umap`, names
    the next command then could not find."""
    truth = ad.read_h5ad(bracketed)
    names = list(truth.obs.columns) + list(truth.obsm.keys())
    for argv in (
        ["view", str(bracketed)],
        ["view", "--tree", "--depth", "1", str(bracketed)],
        ["ls", str(bracketed)],
        ["ls", "--long", str(bracketed)],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, (argv, result.output)
        for name in names:
            assert name in result.stdout, (argv, name, result.stdout)


def test_view_of_an_entry_prints_its_path_and_attributes_as_they_are(bracketed):
    """F042: an attribute value holding `[/oops]` raised Rich's MarkupError,
    so `view` failed on a valid store, and the path lost its `[red]`."""
    result = runner.invoke(app, ["view", str(bracketed), f"obsm/{OBSM_KEY}"])
    assert result.exit_code == 0, result.output
    assert f"obsm/{OBSM_KEY}" in result.stdout
    with h5py.File(bracketed, "r") as f:
        note = f[f"obsm/{OBSM_KEY}"].attrs["note"]
    assert f"note: {note}" in result.stdout


def test_split_names_its_column_and_groups_with_square_brackets_as_they_are(
    bracketed, tmp_path
):
    """F042: `Splitting on obs[batch]` lost `[batch]`, a `[red]b` label was
    coloured and renamed `b`, and an `x[/y]` label crashed the plan."""
    labels = list(ad.read_h5ad(bracketed).obs["batch[1]"].cat.categories)
    result = runner.invoke(
        app, ["split", str(bracketed), "--by", "batch[1]", "-o", str(tmp_path / "out"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "obs[batch[1]]" in result.stderr
    for label in labels:
        assert repr(label) in result.stderr, (label, result.stderr)


def test_messages_name_an_output_whose_path_holds_square_brackets(bracketed, tmp_path):
    """F042: paths are user input too; `Wrote [red]out.h5ad` lost `[red]`."""
    output = tmp_path / "[red]out.h5ad"
    result = runner.invoke(
        app, ["recast", str(bracketed), "X", "-o", str(output), "--dtype", "float64"]
    )
    assert result.exit_code == 0, result.output
    assert ad.read_h5ad(output).X.dtype == np.float64
    assert f"Wrote {output}" in result.stderr


def test_view_of_a_missing_entry_fails_on_stderr_with_exit_1(sample_h5ad_file):
    """F096: the `Error:` line went to stdout with exit 0, so a script took
    it for the entry's description; `ls` already failed properly."""
    result = runner.invoke(app, ["view", str(sample_h5ad_file), "obsm/X_missing"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("Error:")
    assert "obsm/X_missing" in result.stderr


def _run_redirected(argv, tmp_path, env=None):
    """Run the CLI as a pipeline step does: every stream a file, no terminal."""
    out, err = tmp_path / "out.txt", tmp_path / "err.txt"
    clean = {k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "COLUMNS", "NO_COLOR")}
    clean["TERM"] = "xterm-256color"
    clean.update(env or {})
    with open(out, "wb") as stdout, open(err, "wb") as stderr:
        code = subprocess.run(
            [sys.executable, "-c", "from adata.cli import main; main()", *argv],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=clean,
            timeout=120,
        ).returncode
    return code, out.read_text(), err.read_text()


def test_an_error_in_a_log_file_is_one_plain_line_starting_error(sample_h5ad_file, tmp_path):
    """F151: the stderr console forced a terminal, so `2> log` held cursor
    and colour codes before `Error:` and the message wrapped at column 80:
    `grep '^Error:' log` found nothing, against what the pipeline docs say."""
    deep = tmp_path / ("a_rather_long_folder_name_" * 4) / "and_a_long_output_name.h5ad"
    code, out, err = _run_redirected(
        ["subset", str(sample_h5ad_file), "-o", str(deep), "--obs-query", "nope == 1"], tmp_path
    )
    assert code == 1
    assert out == ""
    assert "\x1b" not in err
    errors = [line for line in err.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1, err
    assert "nope" in errors[0]
    assert err.endswith(errors[0] + "\n"), err


def test_output_piped_to_a_file_has_no_terminal_codes(sample_h5ad_file, tmp_path):
    """F151: neither stream holds ANSI codes when it is not a terminal."""
    code, out, err = _run_redirected(["view", "--tree", str(sample_h5ad_file)], tmp_path)
    assert code == 0, err
    assert "\x1b" not in out + err
    assert "obs" in out


def _help_sources():
    """(command path, help text) for every command and option, as written."""
    found = []

    def walk(command, path):
        found.append((path, command.help or ""))
        for param in command.params:
            found.append((path, getattr(param, "help", None) or ""))
        for name, sub in (getattr(command, "commands", None) or {}).items():
            walk(sub, path + [name])

    walk(get_command(app), [])
    return found


def _rendered_help(path):
    result = runner.invoke(app, path + ["--help"], env={"COLUMNS": "400"}, terminal_width=400)
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_convert_help_keeps_the_extras_in_its_install_hints():
    """F160: Rich read `[spatial]` and `[pickle]` as tags, so the hints said
    only "(needs pyadata-cli)", which anyone reading them already has."""
    rendered = _rendered_help(["convert"])
    assert "pyadata-cli[spatial]" in rendered
    assert "pyadata-cli[pickle]" in rendered


def test_every_bracketed_word_in_help_text_survives_rendering():
    """F160, in general: any `[word]` a help string holds reaches the reader."""
    tag = re.compile(r"\\?\[([a-z#/@][^\[\]]*)\]")
    missing = []
    for path, text in _help_sources():
        words = {f"[{m}]" for m in tag.findall(text)}
        if not words:
            continue
        rendered = _rendered_help(path)
        missing += [(path, w) for w in sorted(words) if w not in rendered]
    assert not missing, missing


def _leave_raises(monkeypatch):
    """`main` ends a failed run with os._exit, which would end pytest too."""

    def leave(code):
        raise SystemExit(code)

    monkeypatch.setattr(cli, "_leave", leave)


def test_main_runs_the_arguments_it_is_given(monkeypatch, capsys):
    """F161: `main(argv)` ignored argv and parsed the host's sys.argv, so an
    embedding caller ran some other command, or failed on its arguments."""
    _leave_raises(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["adata", "no-such-command"])
    with pytest.raises(SystemExit) as done:
        cli.main(["--version"])
    assert done.value.code == 0
    assert capsys.readouterr().out.strip() == version("pyadata-cli")


def test_the_deprecated_entry_point_warns_on_stderr_and_runs_its_arguments(
    monkeypatch, capsys
):
    """F161: `main_deprecated(argv)` passed argv on to a `main` that dropped it."""
    _leave_raises(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["h5ad", "no-such-command"])
    with pytest.raises(SystemExit) as done:
        cli.main_deprecated(["--version"])
    assert done.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == version("pyadata-cli")
    assert "'h5ad' command is deprecated" in captured.err


def test_main_exits_with_the_failing_code(monkeypatch, tmp_path):
    """A failed run still leaves through `_leave`, with its exit code."""
    codes = []
    monkeypatch.setattr(cli, "_leave", lambda code: codes.append(code))
    with pytest.raises(SystemExit):
        cli.main(["view", str(tmp_path / "missing.h5ad")])
    assert codes == [2]
