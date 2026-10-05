import pytest

from pxdbench.pxd_configs.eval import eval_configs
from pxdesign.runner.pipeline import detect_use_boltz_filter, detect_use_ptx_filter


class Cfg(dict):
    """Attribute access over a nested dict, like ml_collections."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Cfg(value) if isinstance(value, dict) else value


def test_gate_block_exists_and_is_provisional():
    filters = eval_configs["binder"]["filters"]
    assert "bz_gate_egfr_provisional_v1" in filters
    gate = filters["bz_gate_egfr_provisional_v1"]
    assert gate["bz_n_seeds"] == (">=", 3)
    assert gate["bz_pae_interface_min"] == ("<=", 2.0)
    assert gate["bz_ipsae"] == (">=", 0.5)
    assert "bz_interface_plddt" not in gate, "recorded only, never gated"


def test_gate_uses_only_supported_operators():
    """The new gate needs <= and >=, which the filter fix added."""
    from pxdbench.tasks.base import SUPPORTED_FILTER_OPS

    gate = eval_configs["binder"]["filters"]["bz_gate_egfr_provisional_v1"]
    for metric, (sym, _thres) in gate.items():
        assert sym in SUPPORTED_FILTER_OPS, f"{metric}: {sym!r}"


def test_boltz_tool_defaults():
    boltz = eval_configs["binder"]["tools"]["boltz"]
    assert boltz["seeds_rank"] == 1
    assert boltz["seeds_gate"] == 3
    assert boltz["ranking_key"] == "bz_ipsae"
    assert boltz["compare_ranking_keys"] is False


def test_target_msa_default_is_empty_because_it_is_an_override():
    """MSAs come from the entities in orig_seqs, correctly paired by design
    chain. A configured map here would mispair them."""
    assert eval_configs["binder"]["tools"]["boltz"]["target_msa"] == {}
    assert eval_configs["binder"]["tools"]["boltz"]["a3m_name"] == "non_pairing.a3m"


def test_no_final_rescore_switch():
    """The common batch is mandatory; a switch would restore cross-batch
    ranking silently."""
    assert "final_rescore" not in eval_configs["binder"]["tools"]["boltz"]


def test_num_seqs_default_stays_one():
    """Best-of-8 is opt-in until the milestone shows a gain."""
    assert eval_configs["binder"]["num_seqs"] == 1


def test_eval_boltz_defaults_off():
    assert eval_configs["binder"]["eval_boltz"] is False


def test_detect_use_boltz_filter():
    assert detect_use_boltz_filter(Cfg({"eval": {"binder": {"eval_boltz": True}}}))
    assert not detect_use_boltz_filter(Cfg({"eval": {"binder": {"eval_boltz": False}}}))
    assert not detect_use_boltz_filter(Cfg({"eval": {"binder": {}}}))


def test_eval_boltz_does_not_enable_the_ptx_path():
    """The ptx flag gates a target-only PROTENIX prediction; Boltz must not
    route through it."""
    cfg = Cfg(
        {
            "eval": {
                "binder": {
                    "eval_boltz": True,
                    "eval_protenix": False,
                    "eval_protenix_mini": False,
                }
            }
        }
    )
    assert detect_use_boltz_filter(cfg)
    assert not detect_use_ptx_filter(cfg)


def test_boltz_mode_bypasses_template_heuristic(monkeypatch):
    """use_target_template_or_not runs a Protenix prediction; it must not run."""
    import pxdesign.runner.pipeline as pl

    called = []
    monkeypatch.setattr(
        pl, "use_target_template_or_not", lambda *a, **k: called.append(1) or True
    )
    pipeline = object.__new__(pl.DesignPipeline)
    pipeline.use_ptx_filter = False
    pipeline.use_boltz_filter = True
    assert pipeline.resolve_use_target_template(None, None, 0) is False
    assert called == []


def test_template_heuristic_also_skipped_when_both_flags_set(monkeypatch):
    """Boltz wins: its calibration is template-free."""
    import pxdesign.runner.pipeline as pl

    called = []
    monkeypatch.setattr(
        pl, "use_target_template_or_not", lambda *a, **k: called.append(1) or True
    )
    pipeline = object.__new__(pl.DesignPipeline)
    pipeline.use_ptx_filter = True
    pipeline.use_boltz_filter = True
    assert pipeline.resolve_use_target_template(None, None, 0) is False
    assert called == []
