"""Environment resolution must not sweep the machine to answer one question.

`discover()` probes every candidate interpreter and only then picks the first
that qualifies. Each probe has a 120 s timeout, and any env holding protenix
also pays ~80 s for the fused-LayerNorm build that always fails here and is
therefore never cached. On a machine with many conda envs a campaign launch sat
at `== preflight ==` for minutes having printed nothing - indistinguishable from
a hang, and the reason a working pipeline looked broken.

These tests pin the two properties that fix: the project's own env is tried
first, and a single value is resolved lazily.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pxd_env  # noqa: E402


def test_the_project_env_is_tried_before_any_conda_root():
    """toolenv.py looks in .pxd/envs first; this module must agree with it.

    When the two disagree, a machine with a perfectly good project env still
    pays for a full sweep.
    """
    project = os.path.join(REPO, ".pxd", "envs", "pxd", "bin", "python")
    cands = pxd_env.candidate_pythons()
    if not os.access(project, os.X_OK):
        # Nothing to assert on a checkout that has not run setup.sh; the
        # ordering test below still covers the intent.
        return
    assert project in cands
    conda_idx = [
        i for i, p in enumerate(cands)
        if "/envs/" in p and not p.startswith(os.path.join(REPO, ".pxd"))
    ]
    if conda_idx:
        assert cands.index(project) < min(conda_idx), (
            "the project env must precede every conda root"
        )


def test_resolving_shadowing_probes_no_interpreter(monkeypatch):
    """It is a pure filesystem question. run_campaign.sh asks it on every
    launch, and asking it used to sweep the machine."""
    def explode(*a, **k):
        raise AssertionError("probe() must not be called to answer shadowing")

    monkeypatch.setattr(pxd_env, "probe", explode)
    result = pxd_env.resolve_one("shadowing")
    assert isinstance(result, list)


def test_resolving_one_interpreter_stops_at_the_first_match(monkeypatch):
    """The defect was eager probing: build the whole list, then take [0]."""
    calls = []

    def fake_candidates():
        return ["/first/python", "/second/python", "/third/python"]

    def fake_probe(path, timeout=120):
        calls.append(path)
        return {
            "python": path,
            "diffusion_symbol": True,
            "cuda_usable": True,
            "pytest": True,
            "imports_project": True,
        }

    monkeypatch.setattr(pxd_env, "candidate_pythons", fake_candidates)
    monkeypatch.setattr(pxd_env, "probe", fake_probe)

    assert pxd_env.resolve_one("diffusion_python") == "/first/python"
    assert calls == ["/first/python"], (
        f"probed {len(calls)} interpreters to find the first match: {calls}"
    )


def test_resolving_one_keeps_looking_past_an_unusable_interpreter(monkeypatch):
    """Lazy must not mean 'give up on the first candidate'."""
    def fake_candidates():
        return ["/broken/python", "/good/python", "/never/python"]

    def fake_probe(path, timeout=120):
        usable = path == "/good/python"
        return {
            "python": path,
            "diffusion_symbol": usable,
            "cuda_usable": usable,
            "pytest": usable,
            "imports_project": usable,
        }

    monkeypatch.setattr(pxd_env, "candidate_pythons", fake_candidates)
    monkeypatch.setattr(pxd_env, "probe", fake_probe)
    assert pxd_env.resolve_one("diffusion_python") == "/good/python"


def test_an_unresolvable_value_is_none_not_an_exception(monkeypatch):
    def fake_candidates():
        return ["/a/python"]

    def fake_probe(path, timeout=120):
        return {"python": path, "diffusion_symbol": False, "cuda_usable": False,
                "pytest": False, "imports_project": False}

    monkeypatch.setattr(pxd_env, "candidate_pythons", fake_candidates)
    monkeypatch.setattr(pxd_env, "probe", fake_probe)
    assert pxd_env.resolve_one("diffusion_python") is None


def test_the_launcher_asks_for_shadowing_not_the_full_report():
    """run_campaign.sh used to pipe --json into a reader of info['shadowing'],
    paying for every interpreter on the machine to answer it."""
    src = open(os.path.join(REPO, "scripts", "run_campaign.sh")).read()
    # The invocation is quoted ("$REPO/scripts/pxd_env.py"), so match the flag
    # against the line that runs it rather than a bare concatenation.
    runs = [
        ln for ln in src.splitlines()
        if "pxd_env.py" in ln and not ln.lstrip().startswith("#")
    ]
    shadow_calls = [ln for ln in runs if "--shadowing" in ln]
    json_calls = [ln for ln in runs if "--json" in ln]
    assert shadow_calls, f"no --shadowing call found; pxd_env.py lines: {runs}"
    assert not json_calls, (
        f"the launcher still asks for the full report: {json_calls}"
    )


def test_campaign_defaults_are_set_before_arguments_are_parsed():
    """The launcher silently discarded every argument it was given.

    The defaults block (`INPUT=""; OUT=""; ...`) sat AFTER the parsing loop,
    left behind when the parsing was moved to the top of the script so bad
    flags would be rejected immediately (commit 3a49b79). The result: args were
    parsed, validated by the `-n "$OUT"` guard, and then overwritten with
    empty strings. The run died on `mkdir -p ""` after a full preflight, having
    apparently accepted its arguments. Every launch since that commit was
    affected; it went unseen because runs failed earlier, at boltz resolution.
    """
    src = open(os.path.join(REPO, "scripts", "run_campaign.sh")).read().splitlines()

    def line_of(pred, what):
        hits = [i for i, ln in enumerate(src) if pred(ln)]
        assert hits, f"could not find {what} in run_campaign.sh"
        return hits[0]

    defaults = line_of(lambda ln: ln.startswith('INPUT=""; OUT=""'), "the defaults block")
    parsing = line_of(lambda ln: "while [[ $# -gt 0 ]]" in ln, "the parsing loop")
    assert defaults < parsing, (
        f"defaults at line {defaults + 1} come AFTER the parsing loop at line "
        f"{parsing + 1}, so every parsed argument is overwritten"
    )


def test_campaign_defaults_are_assigned_exactly_once():
    """A second copy left behind anywhere below the parser reintroduces the bug."""
    src = open(os.path.join(REPO, "scripts", "run_campaign.sh")).read().splitlines()
    assigns = [i for i, ln in enumerate(src) if ln.startswith('INPUT=""; OUT=""')]
    assert len(assigns) == 1, f"expected one defaults block, found {len(assigns)}"
