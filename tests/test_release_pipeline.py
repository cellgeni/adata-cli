"""The workflows test what they say, publish only what passed, and hold no
token they do not need.

Three of the four workflows run only on a tag, so a mistake in them shows up
at a release and nowhere earlier. These tests read the workflow files and the
Dockerfile as GitHub and Docker would, and check the properties that matter
at the next tag: which jobs can publish in which run, which hold a write
token, and what each runs. The `workflows` job in tests.yml lints the same
files with actionlint and zizmor.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name):
    workflow = yaml.safe_load((WORKFLOWS / name).read_text())
    # YAML 1.1 reads the bare key `on` as True.
    if True in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _workflows():
    return sorted(path.name for path in WORKFLOWS.glob("*.yml"))


def _evaluate(expression, results, github, inputs):
    """A job's `if:`, for the subset of GitHub's expression syntax used here.

    `results` holds the result of each job it needs. As on GitHub, an
    expression with no status function is implicitly `success() && (...)`.
    """
    expression = " ".join(expression.split())  # `if: |` spans lines
    if expression.startswith("${{"):
        expression = expression[3:-2]
    implicit = not re.search(r"\b(success|failure|always|cancelled)\(\)", expression)
    succeeded = all(result == "success" for result in results.values())

    py = re.sub(
        r"\bneeds\.([\w-]+)\.result\b", lambda m: repr(results[m[1]]), expression
    )
    py = re.sub(r"\bgithub\.(\w+)", lambda m: repr(github[m[1]]), py)
    py = re.sub(r"\binputs\.(\w+)", lambda m: repr(inputs.get(m[1])), py)
    py = py.replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"!(?!=)", " not ", py)
    py = py.replace("always()", "True").replace("success()", repr(succeeded))
    py = py.replace("failure()", "False").replace("cancelled()", "False")
    py = py.replace("startsWith(", "_starts(")
    value = bool(eval(py, {"__builtins__": {}, "_starts": str.startswith}))
    return value and (succeeded or not implicit)


def _run(workflow, event, ref, inputs=None, failing=()):
    """Each job's result in one run, given which jobs fail if they run."""
    jobs = workflow["jobs"]
    github = {
        "event_name": event,
        "ref": ref,
        "ref_type": "tag" if ref.startswith("refs/tags/") else "branch",
        "ref_name": ref.rsplit("/", 1)[-1],
    }
    results = {}

    def result(name):
        if name not in results:
            needs = jobs[name].get("needs", [])
            needs = [needs] if isinstance(needs, str) else needs
            before = {need: result(need) for need in needs}
            runs = _evaluate(jobs[name].get("if", "success()"), before, github, inputs or {})
            results[name] = (
                "skipped" if not runs else "failure" if name in failing else "success"
            )
        return results[name]

    for name in jobs:
        result(name)
    return results


PUBLISHERS = {"publish-testpypi", "publish-pypi", "image"}
TAG = "refs/tags/0.17.5"


@pytest.mark.parametrize(
    "event, ref, target, failing, published",
    [
        ("push", TAG, None, (), {"publish-pypi", "image"}),
        ("workflow_dispatch", TAG, "testpypi", (), {"publish-testpypi"}),
        ("workflow_dispatch", TAG, "pypi", (), {"publish-pypi", "image"}),
        ("workflow_dispatch", "refs/heads/main", "testpypi", (), {"publish-testpypi"}),
        ("workflow_dispatch", "refs/heads/main", "pypi", (), set()),
        ("push", TAG, None, ("test",), set()),
        ("push", TAG, None, ("check-version",), set()),
        ("push", TAG, None, ("build",), set()),
        ("push", TAG, None, ("publish-pypi",), set()),
    ],
    ids=[
        "tag", "testpypi-dispatch-from-tag", "pypi-dispatch-from-tag",
        "testpypi-dispatch-from-branch", "pypi-dispatch-from-branch",
        "tests-fail", "tag-is-not-the-version", "build-fails", "pypi-upload-fails",
    ],
)
def test_each_run_publishes_exactly_what_passed(event, ref, target, failing, published):
    """Up to 0.17.4 a TestPyPI dispatch from a tag also published to PyPI,
    a PyPI dispatch from a branch published whatever pyproject said, and the
    image was pushed, and :latest moved, on any tag, tests or no tests.
    """
    inputs = {"target": target} if target else {}
    results = _run(_load("publish.yml"), event, ref, inputs, failing)
    assert {
        job for job in PUBLISHERS if results.get(job) == "success"
    } == published


def test_the_image_is_pushed_only_from_publish():
    """quay-on-tag.yml ran on every tag by itself until 0.17.4."""
    assert set(_load("quay-on-tag.yml")["on"]) == {"workflow_call"}
    image = _load("publish.yml")["jobs"]["image"]
    assert image["uses"] == "./.github/workflows/quay-on-tag.yml"
    assert "inherit" not in str(image.get("secrets")), "pass Quay's secrets by name"


@pytest.mark.parametrize("name", _workflows())
def test_every_job_says_what_its_token_may_do(name):
    """The repository's default token can write, so a job that names no
    permissions gets write access to everything. publish.yml named none
    until 0.17.5."""
    workflow = _load(name)
    for job_name, job in workflow["jobs"].items():
        assert "permissions" in job or "permissions" in workflow, f"{name}: {job_name}"


INSTALLS = re.compile(r"(^|[\s;&|(])(uv|uvx|pip3?|pipx|conda|npm|curl|wget)\s")


@pytest.mark.parametrize("name", _workflows())
def test_a_job_that_can_push_installs_nothing(name):
    """Up to 0.17.4 the benchmark job held contents: write while it installed
    the newest anndata and scanpy from PyPI and ran them."""
    workflow = _load(name)
    default = workflow.get("permissions") or {}
    for job_name, job in workflow["jobs"].items():
        permissions = job.get("permissions", default)
        if not isinstance(permissions, dict) or permissions.get("contents") != "write":
            continue
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            assert not uses.startswith(("astral-sh/setup-uv", "actions/setup-python")), (
                f"{name}: {job_name} sets up an installer"
            )
            assert not INSTALLS.search(step.get("run", "")), (
                f"{name}: {job_name} installs in {step.get('name')!r}"
            )


#: Contexts whose values someone other than the workflow's author chooses:
#: a tag or branch name, an input, the event's payload, another job's output.
UNTRUSTED = re.compile(
    r"\$\{\{[^}]*\b(github\.(ref|ref_name|head_ref|event)|inputs\.|needs\.|steps\.)"
)


@pytest.mark.parametrize("name", _workflows())
def test_no_shell_script_interpolates_an_untrusted_value(name):
    """`${{ }}` is pasted into the script before the shell sees it, so a tag
    named `$(...)` ran as code. benchmark.yml did this with the tag until
    0.17.5; pass such values through `env:` instead."""
    for job_name, job in _load(name)["jobs"].items():
        for step in job.get("steps", []):
            found = UNTRUSTED.search(step.get("run", ""))
            assert not found, f"{name}: {job_name}: {found and found[0]}"


USES = re.compile(r"^\s*-?\s*uses:\s*(\S+)(.*)$", re.M)


@pytest.mark.parametrize("name", _workflows())
def test_every_action_is_pinned_to_a_commit(name):
    """A tag such as `v4` can be moved to other code by whoever controls the
    action, and runs in jobs that hold secrets or publish."""
    for action, rest in USES.findall((WORKFLOWS / name).read_text()):
        if action.startswith("./"):
            continue
        ref = action.rpartition("@")[2]
        assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{name}: {action}"
        assert re.match(r"\s+# v\d", rest), f"{name}: {action} has no version comment"


def test_each_python_leg_runs_its_own_python():
    """uv takes its interpreter from .python-version (3.12) before the one on
    PATH, so until 0.17.5 the py3.13 leg ran the suite on 3.12. UV_PYTHON
    outranks the file, and the leg checks what it got before testing."""
    job = _load("tests.yml")["jobs"]["tests"]
    assert job["env"]["UV_PYTHON"] == "${{ matrix.python-version }}"
    pinned = (ROOT / ".python-version").read_text().strip()
    assert any(v != pinned for v in job["strategy"]["matrix"]["python-version"])
    checks = [s for s in job["steps"] if "$UV_PYTHON" in s.get("run", "")]
    assert checks and job["steps"].index(checks[0]) < next(
        i for i, s in enumerate(job["steps"]) if "pytest" in s.get("run", "")
    )


def test_the_image_is_built_from_pinned_inputs_and_runs_unprivileged():
    """Until 0.17.5 the base was a floating tag, duckdb was unzipped into
    /usr/local/bin unchecked, and the image ran as root."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    instructions = re.findall(r"^([A-Z]+)\s+(.*)$", dockerfile, re.M)

    base = [arg for op, arg in instructions if op == "FROM"]
    assert base and all(re.search(r"@sha256:[0-9a-f]{64}$", b) for b in base)

    download = dockerfile.index("duckdb_cli-linux")
    assert dockerfile.index("sha256sum -c", download) < dockerfile.index("unzip", download)
    for arch in ("AMD64", "AARCH64"):
        assert re.search(rf"^ARG DUCKDB_SHA256_{arch}=[0-9a-f]{{64}}$", dockerfile, re.M)

    users = [arg for op, arg in instructions if op == "USER"]
    assert users and users[-1].split(":")[0] not in {"0", "root"}
    last_run = max(i for i, (op, _) in enumerate(instructions) if op == "RUN")
    assert instructions.index(("USER", users[-1])) > last_run, (
        "a RUN after USER would fail for want of root"
    )
