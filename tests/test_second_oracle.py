"""funnel/oracles.py without a GPU: pure helpers, the PAE -> ipSAE parser, and the AF3 runner driven against a stub run_alphafold.py."""
import json, sys, threading, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "funnel")); sys.path.insert(0, str(REPO))
import common, oracles  # noqa: E402

TARGET = "ACDEFGHIKLMNPQRSTVWY" * 2          # 40 aa
BINDER = "MKKLEELLKKAEELLKKHGDEE"            # 22 aa


# ----------------------------------------------------------------------------- pure helpers
def test_af3_name_keeps_safe_ids_and_hashes_the_rest():
    assert oracles.af3_name("d12_s3.1-a") == "d12_s3.1-a"
    n1, n2 = oracles.af3_name("Design A/7"), oracles.af3_name("Design A/8")
    assert n1 != n2 and n1.startswith("x") and len(n1) == 13 and n1 == oracles.af3_name("Design A/7")


def test_sanitize_a3m_drops_what_af3_cannot_parse(tmp_path):
    q = "ACDEFGHIKL"
    src = tmp_path / "in.a3m"
    src.write_bytes((f">query\n{q}\n>ok\nACDEFGHIKL\n>with_insertion\nACDEfFGHIKL\n>gappy\nA-DEFGHIKL\n"
                     ">truncated\nACDEFG\n>nul\nACDE\x00FGHIKL\n>bad_char\nACDE*FGHIK\n>too_long\nACDEFGHIKLAA\n").encode())
    st = oracles.sanitize_a3m(src, tmp_path / "out.a3m", q)
    kept = [l for l in (tmp_path / "out.a3m").read_text().splitlines() if l.startswith(">")]
    assert kept == [">query", ">ok", ">with_insertion", ">gappy"] and st == dict(total=8, kept=4, dropped=4)


def test_sanitize_a3m_refuses_a_file_whose_first_record_is_not_the_query(tmp_path):
    (tmp_path / "x.a3m").write_text(">q\nAAAAAAAAAA\n")
    with pytest.raises(ValueError): oracles.sanitize_a3m(tmp_path / "x.a3m", tmp_path / "y.a3m", "CCCCCCCCCC")


def test_af3_input_injects_the_msa_and_forbids_templates(tmp_path):
    msa = tmp_path / "t.a3m"; msa.write_text(">q\n" + TARGET + "\n")
    j = oracles.af3_input("d1", TARGET, msa, BINDER, 7)
    a, b = j["sequences"][0]["protein"], j["sequences"][1]["protein"]
    assert j["modelSeeds"] == [7] and j["dialect"] == "alphafold3" and a["id"] == "A" and b["id"] == "B"
    assert a["unpairedMsaPath"] == str(msa.resolve()) and a["templates"] == [] and b["templates"] == []
    assert b["unpairedMsa"] == f">query\n{BINDER}\n" and a["sequence"] == TARGET and b["sequence"] == BINDER


def test_af3_command_skips_the_data_pipeline_and_pins_one_device():
    c = oracles.af3_command("py", "/af3", "/x/in.json", "/x/out", "/m", "/cache", 5)
    assert c[0] == "py" and c[1] == "/af3/run_alphafold.py" and "--norun_data_pipeline" in c and "--gpu_device=0" in c and "--num_diffusion_samples=5" in c


def test_af3_env_isolates_one_gpu_and_does_not_preallocate():
    e = oracles.af3_env("3"); assert e["CUDA_VISIBLE_DEVICES"] == "3" and e["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" and e["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"


def test_af3_config_names_the_missing_variables(monkeypatch):
    for k in ("PXD_AF3_PYTHON", "PXD_AF3_DIR", "PXD_AF3_MODELS"): monkeypatch.delenv(k, raising=False)
    with pytest.raises(FileNotFoundError, match="PXD_AF3_PYTHON"): oracles.af3_config()


# ----------------------------------------------------------------------------- parsing an AF3 output folder
def _write_sample(d, name, seed, k, pae, tc, iptm, rank):
    sd = d / f"seed-{seed}_sample-{k}"; sd.mkdir(parents=True)
    json.dump({"pae": np.asarray(pae).tolist(), "token_chain_ids": tc}, open(sd / f"{name}_seed-{seed}_sample-{k}_confidences.json", "w"))
    json.dump({"iptm": iptm, "ranking_score": rank}, open(sd / f"{name}_seed-{seed}_sample-{k}_summary_confidences.json", "w"))


def test_parse_af3_result_is_the_sample_mean_of_the_funnel_ipsae(tmp_path):
    nt, nb = len(TARGET), len(BINDER); tc = ["A"] * nt + ["B"] * nb; d = tmp_path / "d1"
    rng = np.random.default_rng(0); paes = []
    for k in range(3):
        p = rng.uniform(0.5, 25, (nt + nb, nt + nb)); paes.append(p); _write_sample(d, "d1", 1, k, p, tc, 0.5 + 0.1 * k, 0.4 + 0.1 * k)
    (d / "d1_model.cif").write_text("data_x\n")
    r = oracles.parse_af3_result(d, "d1", nt)
    lo = [common.ipsae(p, nt, nb)[0] for p in paes]; hi = [common.ipsae(p, nt, nb)[1] for p in paes]
    assert r["ok"] and r["n_samples"] == 3
    assert r["ipsae"] == pytest.approx(np.mean(lo)) and r["ipsae_max"] == pytest.approx(np.mean(hi))
    assert r["paemin"] == pytest.approx(np.mean([common.pae_interface_min(p, nt) for p in paes]))
    assert r["iptm"] == pytest.approx(0.6) and r["rank"] == pytest.approx(0.6) and r["cif"].endswith("d1_model.cif")


def test_parse_af3_result_refuses_a_mismatched_target_and_reports_empty_folders(tmp_path):
    nt, nb = len(TARGET), len(BINDER); _write_sample(tmp_path / "d", "d", 1, 0, np.full((nt + nb,) * 2, 1.0), ["A"] * nt + ["B"] * nb, 0.5, 0.5)
    (tmp_path / "d" / "d_model.cif").write_text("x")
    with pytest.raises(ValueError, match="expected"): oracles.parse_af3_result(tmp_path / "d", "d", nt + 1)
    assert oracles.parse_af3_result(tmp_path / "missing", "m", nt) == dict(ok=False)


# ----------------------------------------------------------------------------- the runner, against a stub AF3
STUB = r'''
import json, os, sys, time
a = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
J = json.load(open(a["json_path"])); name = J["name"]
open(os.path.join(os.path.dirname(a["output_dir"]), "calls.log"), "a").write(json.dumps(dict(name=name, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
     prealloc=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"), flags=[x for x in sys.argv[1:] if "=" not in x])) + "\n")
time.sleep(float(os.environ.get("STUB_SLEEP_" + str(os.environ.get("CUDA_VISIBLE_DEVICES")), "0.05")))
if "fail" in name: sys.exit(3)
nt = len(J["sequences"][0]["protein"]["sequence"]); nb = len(J["sequences"][1]["protein"]["sequence"]); n = nt + nb
od = os.path.join(a["output_dir"], name); os.makedirs(od, exist_ok=True)
for k in range(int(a["num_diffusion_samples"])):
    sd = os.path.join(od, f"seed-1_sample-{k}"); os.makedirs(sd, exist_ok=True)
    json.dump(dict(pae=[[0.5] * n for _ in range(n)], token_chain_ids=["A"] * nt + ["B"] * nb), open(os.path.join(sd, f"{name}_seed-1_sample-{k}_confidences.json"), "w"))
    json.dump(dict(iptm=0.8, ranking_score=0.7 + 0.01 * k), open(os.path.join(sd, f"{name}_seed-1_sample-{k}_summary_confidences.json"), "w"))
open(os.path.join(od, f"{name}_model.cif"), "w").write("data_stub\n")
'''


@pytest.fixture
def stub_af3(tmp_path, monkeypatch):
    d = tmp_path / "af3dir"; d.mkdir(); (d / "run_alphafold.py").write_text(STUB); (tmp_path / "models").mkdir()
    monkeypatch.setenv("PXD_AF3_PYTHON", sys.executable); monkeypatch.setenv("PXD_AF3_DIR", str(d)); monkeypatch.setenv("PXD_AF3_MODELS", str(tmp_path / "models"))
    msa = tmp_path / "target.a3m"; msa.write_text(f">query\n{TARGET}\n>hit\n{TARGET}\n")
    return dict(seq=TARGET, msa=str(msa)), tmp_path


def _designs(n, extra=()):
    return pd.DataFrame({"id": [f"d{i}" for i in range(n)] + list(extra), "seq": [BINDER] * (n + len(extra))})


def _calls(root):
    f = root / "calls.log"; return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def test_af3_fold_runs_every_design_isolated_on_its_gpu_and_returns_the_v2_columns(stub_af3):
    t, root = stub_af3; out = root / "run" / "af3"
    d = oracles.af3_fold(_designs(6), t, out, seed=1, gpus=["0", "1"], n_samples=3)
    assert list(d.columns) == ["id", "v2_ok", "v2_ipsae", "v2_ipsae_max", "v2_iptm", "v2_rank", "v2_paemin", "v2_cif"]
    assert d.v2_ok.all() and (d.v2_ipsae > 0.8).all() and np.allclose(d.v2_iptm, 0.8) and np.allclose(d.v2_rank, 0.72)
    calls = _calls(root / "run" / "af3"); assert len(calls) == 6 and {c["gpu"] for c in calls} == {"0", "1"} and {c["prealloc"] for c in calls} == {"false"}
    assert all("--norun_data_pipeline" in c["flags"] for c in calls)
    assert d.attrs["gpu_seconds"] > 0 and (out / "target_msa.a3m").exists()


def test_af3_fold_resumes_and_isolates_a_failing_design(stub_af3):
    t, root = stub_af3; out = root / "run" / "af3"
    d = oracles.af3_fold(_designs(3, extra=("des_fail",)), t, out, gpus=["0"], n_samples=2)
    assert d.set_index("id").v2_ok.to_dict() == {"d0": True, "d1": True, "d2": True, "des_fail": False}
    assert [f[0] for f in d.attrs["failures"]] == ["des_fail"] and d.set_index("id").loc["des_fail"].isna()[["v2_ipsae", "v2_cif"]].all()
    n_before = len(_calls(root / "run" / "af3"))
    d2 = oracles.af3_fold(_designs(3, extra=("des_fail",)), t, out, gpus=["0"], n_samples=2)
    assert len(_calls(root / "run" / "af3")) - n_before == 1                                   # only the failed design is retried
    assert d2.v2_ok.sum() == 3


def test_a_shared_queue_gives_the_faster_gpu_proportionally_more_designs(stub_af3, monkeypatch):
    """GPU '0' is 8x faster than GPU '1' in the stub; with a shared queue it takes most of the work without any hand-set weights (the 2:1:1:1 case)."""
    t, root = stub_af3; monkeypatch.setenv("STUB_SLEEP_0", "0.02"); monkeypatch.setenv("STUB_SLEEP_1", "0.6")
    oracles.af3_fold(_designs(12), t, root / "run" / "af3", gpus=["0", "1"], n_samples=1)
    n = pd.Series([c["gpu"] for c in _calls(root / "run" / "af3")]).value_counts()
    assert n["0"] >= 2 * n.get("1", 0) and n.sum() == 12


def test_a_held_gpu_waits_for_its_event(stub_af3):
    """The GPU that Boltz-2 is using must not start AF3 work until Boltz-2 releases it."""
    t, root = stub_af3; ev = threading.Event(); threading.Timer(0.5, ev.set).start(); t0 = time.time()
    d = oracles.af3_fold(_designs(1), t, root / "run" / "af3", gpus=["1"], n_samples=1, hold={"1": ev})
    assert time.time() - t0 >= 0.45 and d.v2_ok.all()
