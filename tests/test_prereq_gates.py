"""The pipeline's startup gates assumed AF2 and Protenix always run.

Both were unconditional and both aborted a Boltz-only run before it generated
anything: check_tool_weights exit(1)'d demanding ~700 MB of AF2 parameters it
would never load, and download_inference_cache tried to fetch Protenix weights
through a broken symlink and raised FileNotFoundError on the absent target.

Found by actually running the pipeline, not by the unit tests.
"""
import pytest

from pxdesign.utils.infer import download_inference_cache
from pxdesign.utils.pipeline import check_tool_weights


class Cfg(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Cfg(value) if isinstance(value, dict) else value


def _binder(**over):
    binder = {
        "eval_complex": False,
        "eval_binder_monomer": False,
        "eval_protenix": False,
        "eval_protenix_mini": False,
    }
    binder.update(over)
    return Cfg({"eval": {"binder": binder}})


def test_tool_weight_check_skipped_when_af2_is_off(capsys, monkeypatch):
    monkeypatch.delenv("TOOL_WEIGHTS_ROOT", raising=False)
    # must NOT raise even with TOOL_WEIGHTS_ROOT unset
    check_tool_weights(_binder())
    assert "Skipping tool-weight check" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["eval_complex", "eval_binder_monomer"])
def test_tool_weight_check_still_runs_when_af2_is_on(flag, monkeypatch):
    monkeypatch.delenv("TOOL_WEIGHTS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="TOOL_WEIGHTS_ROOT"):
        check_tool_weights(_binder(**{flag: True}))


def test_tool_weight_check_unchanged_without_configs(monkeypatch):
    """Legacy callers pass nothing and must keep the old strict behaviour."""
    monkeypatch.delenv("TOOL_WEIGHTS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="TOOL_WEIGHTS_ROOT"):
        check_tool_weights()


def _cache_cfg(tmp_path, **over):
    """A config whose cache files and main checkpoint all already exist, so the
    only thing left for download_inference_cache to do is the Protenix loop."""
    cfg = _binder(**over)
    cfg["model_name"] = "pxdesign_v0.1.0"
    cfg["load_checkpoint_dir"] = str(tmp_path)
    caches = {}
    for name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_text("stub")
        caches[name] = str(path)
    cfg["data"] = Cfg(caches)
    # the tail of download_inference_cache assigns into these
    cfg["eval"]["binder"]["tools"] = {"ptx": {}, "ptx_mini": {}}
    (tmp_path / "pxdesign_v0.1.0.pt").write_text("stub")
    return cfg


def test_protenix_download_skipped_when_protenix_is_off(tmp_path, monkeypatch, caplog):
    """A broken checkpoint symlink reads as missing; without this skip the
    download is attempted and dies on the symlink's absent target.

    urlretrieve is patched rather than download_from_url, which is a NESTED
    function and so unpatchable - a detail that made the first version of this
    test fail.
    """
    import urllib.request

    # a BROKEN symlink, exactly like release_data/checkpoint's dead links
    (tmp_path / "protenix_base_default_v0.5.0.pt").symlink_to("/nonexistent/x.pt")

    calls = []
    monkeypatch.setattr(
        urllib.request, "urlretrieve", lambda *a, **k: calls.append(a[:1])
    )
    import logging

    with caplog.at_level(logging.INFO):
        download_inference_cache(_cache_cfg(tmp_path))
    assert calls == [], f"nothing should be downloaded, got {calls}"
    assert "Skipping Protenix checkpoint download" in caplog.text


def test_protenix_download_attempted_when_protenix_is_on(tmp_path, monkeypatch):
    import urllib.request

    calls = []

    def _fake(url, filename=None, **kw):
        calls.append(url)
        if filename:
            open(filename, "w").write("stub")

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake)
    # check_weight would reject the stub, so stop at the first attempt
    with pytest.raises(Exception):
        download_inference_cache(_cache_cfg(tmp_path, eval_protenix=True))
    assert calls, "Protenix checkpoints must still be fetched when enabled"
    assert any("protenix" in str(u) for u in calls), calls
