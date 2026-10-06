"""The blocker from review: nothing called Boltz at the task level.

BinderTask.run branched only on eval_complex / eval_binder_monomer /
eval_protenix_mini / eval_protenix, and BaseTask.get_ptx constructs the backend
with cfg=self.cfg.tools.ptx or tools.ptx_mini - never tools.boltz. So resolving
BoltzBackend through $PXDBENCH_BACKEND gave it Protenix diffusion config and only
ran it when eval_protenix_mini was set.
"""
import pytest
pytest.importorskip("biotite", reason="needs the full pxd environment (biotite)")

from pxdbench.tasks.base import BaseTask
from pxdbench.tasks.binder import BinderTask


class Recorder:
    """Stands in for BoltzBackend, recording the cfg it was built with."""

    instances = []

    def __init__(self, cfg=None, device=None, **kw):
        self.cfg = cfg
        self.device = device
        self.calls = []
        Recorder.instances.append(self)

    def prepare_json(self, pdb_dir, data_list, **kw):
        self.calls.append(("prepare_json", len(data_list), kw.get("dump_dir")))
        return "manifest.json"

    def predict(self, input_json_path, data_list=None, **kw):
        self.calls.append(("predict", len(data_list or []), kw.get("dump_dir")))
        for item in data_list or []:
            item["bz_ipsae"] = 0.7
            item["bz_status"] = "ok"
            item["bz_n_seeds"] = 3
        return {}


class _Cfg(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _Cfg(value) if isinstance(value, dict) else value

    def get(self, name, default=None):
        value = dict.get(self, name, default)
        return _Cfg(value) if isinstance(value, dict) else value


@pytest.fixture(autouse=True)
def _clear():
    Recorder.instances = []


def _task(monkeypatch, **cfg_over):
    """A BinderTask with its heavy __init__ bypassed."""
    import pxdbench.tasks.base as base

    monkeypatch.setattr(base, "get_backend", lambda spec=None: Recorder)
    task = object.__new__(BinderTask)
    task.task_type = "binder"
    task.task_name = "t"
    task.device_id = -1
    task.seed = 1
    task.pdb_dir = "/pdbs"
    task.out_dir = "/out"
    task.binder_chains = ["C"]
    task.cond_chains = ["A", "B"]
    task._ptx_inst = None
    task._ptx_mini_inst = None
    task._boltz_inst = None
    task.backend_spec = None
    cfg = {
        "tools": {
            "boltz": {"seeds_rank": 1, "seeds_gate": 3, "ranking_key": "bz_ipsae"},
            "ptx": {"N_sample": 1},
            "ptx_mini": {"N_sample": 1},
        },
        "eval_boltz": True,
    }
    cfg.update(cfg_over)
    task.cfg = _Cfg(cfg)
    task.orig_seqs = [
        {
            "proteinChain": {
                "sequence": "MK",
                "count": 1,
                "label_asym_id": ["A0"],
                "msa": {"precomputed_msa_dir": "/cache/a/0"},
            }
        }
    ]
    return task


def _patch_boltz_class(monkeypatch):
    """get_boltz imports BoltzBackend directly, so patch it at its source."""
    import pxdbench.tools.boltz.backend as backend_mod

    monkeypatch.setattr(backend_mod, "BoltzBackend", Recorder)
    return Recorder

def test_get_boltz_receives_tools_boltz_not_tools_ptx(monkeypatch):
    """The whole point: get_ptx would hand it Protenix diffusion config."""
    task = _task(monkeypatch)
    backend = task.get_boltz()
    assert backend.cfg["seeds_gate"] == 3
    assert "N_sample" not in backend.cfg


def test_get_boltz_is_cached(monkeypatch):
    task = _task(monkeypatch)
    _patch_boltz_class(monkeypatch)
    assert task.get_boltz() is task.get_boltz()
    assert len(Recorder.instances) == 1


def test_get_boltz_is_separate_from_get_ptx(monkeypatch):
    task = _task(monkeypatch)
    boltz = task.get_boltz()
    ptx = task.get_ptx(is_large=True)
    assert boltz is not ptx
    assert "N_sample" in ptx.cfg


def test_boltz_predict_calls_prepare_then_predict(monkeypatch):
    task = _task(monkeypatch)
    _patch_boltz_class(monkeypatch)
    data_list = [{"name": "bb1", "seq_idx": 0, "sequence": "AAAA"}]
    task.boltz_predict(data_list, orig_seqs=task.orig_seqs)
    backend = Recorder.instances[-1]
    assert [c[0] for c in backend.calls] == ["prepare_json", "predict"]
    assert data_list[0]["bz_ipsae"] == 0.7


def test_boltz_predict_uses_its_own_dump_dir(monkeypatch):
    """Must not collide with ptx_pred / ptx_mini_pred."""
    task = _task(monkeypatch)
    _patch_boltz_class(monkeypatch)
    task.boltz_predict([{"name": "bb1", "seq_idx": 0, "sequence": "A"}],
                       orig_seqs=task.orig_seqs)
    dumps = {c[2] for c in Recorder.instances[-1].calls}
    assert dumps == {"/out/boltz_pred"}


def test_run_invokes_boltz_when_enabled(monkeypatch):
    """Guards the blocker: without a branch in run(), Boltz is never called."""
    task = _task(monkeypatch)
    task.eval_boltz = True
    task.eval_complex = False
    task.eval_binder_monomer = False
    task.eval_protenix_mini = False
    task.eval_protenix = False
    task.eval_diversity = False

    called = []
    monkeypatch.setattr(
        BinderTask, "design_sequence",
        lambda self, verbose=True: [
            {"name": "bb1", "seq_idx": 0, "sequence": "AAAA"}
        ],
    )
    monkeypatch.setattr(BinderTask, "check_results", lambda self, r: None)
    monkeypatch.setattr(BinderTask, "cal_secondary", lambda self, r, c=None: None)
    monkeypatch.setattr(BinderTask, "cal_diversity", lambda self, **kw: None)
    monkeypatch.setattr(
        BaseTask, "boltz_predict",
        lambda self, data_list, orig_seqs=None: called.append(len(data_list)),
    )
    monkeypatch.setattr(
        BinderTask, "summary_from_df", lambda self, df, other_metrics=None: {}
    )
    monkeypatch.setattr(
        "pxdbench.tasks.binder.save_eval_results",
        lambda *a, **k: ("s.csv", "s.json"),
    )
    task.cfg = _Cfg(dict(task.cfg, filters={}))
    task.sample_fn = "s.csv"
    task.summary_fn = "s.json"
    task.run()
    assert called == [1], "run() did not call boltz_predict"


def test_run_skips_boltz_when_disabled(monkeypatch):
    task = _task(monkeypatch)
    task.eval_boltz = False
    task.eval_complex = False
    task.eval_binder_monomer = False
    task.eval_protenix_mini = False
    task.eval_protenix = False
    task.eval_diversity = False

    called = []
    monkeypatch.setattr(
        BinderTask, "design_sequence",
        lambda self, verbose=True: [
            {"name": "bb1", "seq_idx": 0, "sequence": "AAAA"}
        ],
    )
    monkeypatch.setattr(BinderTask, "check_results", lambda self, r: None)
    monkeypatch.setattr(BinderTask, "cal_secondary", lambda self, r, c=None: None)
    monkeypatch.setattr(BinderTask, "cal_diversity", lambda self, **kw: None)
    monkeypatch.setattr(
        BaseTask, "boltz_predict",
        lambda self, data_list, orig_seqs=None: called.append(len(data_list)),
    )
    monkeypatch.setattr(
        BinderTask, "summary_from_df", lambda self, df, other_metrics=None: {}
    )
    monkeypatch.setattr(
        "pxdbench.tasks.binder.save_eval_results",
        lambda *a, **k: ("s.csv", "s.json"),
    )
    task.cfg = _Cfg(dict(task.cfg, filters={}))
    task.sample_fn = "s.csv"
    task.summary_fn = "s.json"
    task.run()
    assert called == []


def test_eval_boltz_is_read_in_init():
    """Reads from cfg, defaulting off."""
    assert _Cfg({"eval_boltz": True}).get("eval_boltz", False) is True
    assert _Cfg({}).get("eval_boltz", False) is False


def test_get_boltz_does_not_use_the_global_ptx_factory(monkeypatch):
    """PXDBENCH_BACKEND is GLOBAL. Routing get_boltz through ptx_factory meant
    pointing it at BoltzBackend also replaced the Protenix backend, so the
    Protenix stage silently ran Boltz with Protenix's config and emitted no
    ptx_* columns. Found by running the pipeline with both stages enabled.
    """
    from pxdbench.tools.boltz.backend import BoltzBackend

    task = _task(monkeypatch)          # patches get_backend -> Recorder
    boltz = task.get_boltz()
    assert isinstance(boltz, BoltzBackend), type(boltz)
    assert not isinstance(boltz, Recorder)
    # the ptx slots still resolve through the registry, so Protenix is intact
    assert isinstance(task.get_ptx(is_large=True), Recorder)


def test_boltz_backend_class_override(monkeypatch):
    """A dotted spec in tools.boltz.backend_class swaps the class."""
    task = _task(monkeypatch)
    task.cfg["tools"]["boltz"]["backend_class"] = (
        "pxdbench.tools.ptx.ptx:ProtenixFilter"
    )
    from pxdbench.tools.ptx.ptx import ProtenixFilter

    captured = {}
    monkeypatch.setattr(
        ProtenixFilter, "__init__",
        lambda self, cfg=None, device=None, **kw: captured.update(cfg=cfg),
    )
    got = task.get_boltz()
    assert isinstance(got, ProtenixFilter)
    assert captured["cfg"]["seeds_gate"] == 3, "still gets tools.boltz"
