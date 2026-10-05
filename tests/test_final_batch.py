import pandas as pd
import pytest

from pxdbench.tools.boltz.final_batch import rescore_shortlist

_ORIG_SEQS = [
    {"proteinChain": {"sequence": "MKTAYIAK", "count": 1,
                      "label_asym_id": ["A0"],
                      "msa": {"precomputed_msa_dir": "/cache/chainB/0"}}},
    {"proteinChain": {"sequence": "QRSTVWYC", "count": 1,
                      "label_asym_id": ["B0"],
                      "msa": {"precomputed_msa_dir": "/cache/chainD/0"}}},
]


class FakeBackend:
    """Stands in for BoltzBackend.

    Exposes score_flat, NOT predict: the two-tier predict would re-select
    winners and leave the rest at one seed, which is the defect this task
    exists to avoid.
    """

    def __init__(self):
        self.seen = None
        self.flat_calls = []
        self.batch_id = "final-1"

    def prepare_json(self, pdb_dir, data_list, **kw):
        self.seen = [dict(d) for d in data_list]
        return "manifest.json"

    def score_flat(self, entries, dump_dir, seeds):
        self.flat_calls.append({"n": len(entries), "seeds": seeds})
        return {
            entry["sample_name"]: {
                "bz_ipsae": 0.5 + i * 0.1,
                "bz_ipdae": 0.4 + i * 0.1,
                "bz_n_seeds": seeds,
                "bz_status": "ok",
                "bz_batch_id": self.batch_id,
                "bz_final_batch_id": self.batch_id,
            }
            for i, entry in enumerate(entries)
        }

    def predict(self, *a, **kw):  # pragma: no cover - must never be called
        raise AssertionError(
            "rescore_shortlist must use score_flat, not the two-tier predict"
        )


def _shortlist():
    return pd.DataFrame(
        [
            {"name": "bb1", "seq_idx": 0, "sequence": "AAAA",
             "bz_ipsae": 0.9, "bz_final_batch_id": "shard-1"},
            {"name": "bb2", "seq_idx": 0, "sequence": "CCCC",
             "bz_ipsae": 0.8, "bz_final_batch_id": "shard-2"},
        ]
    )


def test_rescore_gives_one_shared_final_batch_id(tmp_path):
    out = rescore_shortlist(
        _shortlist(), FakeBackend(), str(tmp_path), _ORIG_SEQS
    )
    assert out["bz_final_batch_id"].nunique() == 1


def test_rescore_replaces_the_per_shard_scores(tmp_path):
    out = rescore_shortlist(
        _shortlist(), FakeBackend(), str(tmp_path), _ORIG_SEQS
    )
    assert list(out["bz_ipsae"]) != [0.9, 0.8]


def test_rescore_scores_every_shortlist_row(tmp_path):
    backend = FakeBackend()
    rescore_shortlist(_shortlist(), backend, str(tmp_path), _ORIG_SEQS)
    assert {d["name"] for d in backend.seen} == {"bb1", "bb2"}


def test_every_row_gets_the_same_seed_depth(tmp_path):
    """Equal-depth scoring is the whole point: the two-tier predict would give
    three seeds only to its own newly selected winners."""
    backend = FakeBackend()
    out = rescore_shortlist(
        _shortlist(), backend, str(tmp_path), _ORIG_SEQS, seeds=3
    )
    assert set(out["bz_n_seeds"]) == {3}
    assert backend.flat_calls == [{"n": 2, "seeds": 3}]


def test_uses_score_flat_not_predict(tmp_path):
    """FakeBackend.predict raises; reaching it is the failure."""
    rescore_shortlist(_shortlist(), FakeBackend(), str(tmp_path), _ORIG_SEQS)


def test_batch_size_is_bounded_by_the_shortlist(tmp_path):
    """Called AFTER triage, so the batch is the shortlist, not the pooled rows."""
    backend = FakeBackend()
    big = pd.concat([_shortlist()] * 10, ignore_index=True)
    big["seq_idx"] = range(len(big))
    rescore_shortlist(big.head(4), backend, str(tmp_path), _ORIG_SEQS)
    assert backend.flat_calls[0]["n"] == 4


def test_rescore_includes_the_anchor_when_configured(tmp_path):
    backend = FakeBackend()
    rescore_shortlist(
        _shortlist(), backend, str(tmp_path), _ORIG_SEQS,
        anchor_sequence="WWWW",
    )
    assert any(d["sequence"] == "WWWW" for d in backend.seen)


def test_anchor_row_is_not_returned_as_a_design(tmp_path):
    out = rescore_shortlist(
        _shortlist(), FakeBackend(), str(tmp_path), _ORIG_SEQS,
        anchor_sequence="WWWW",
    )
    assert "WWWW" not in set(out["sequence"])
    assert len(out) == 2


def test_empty_shortlist_returns_empty(tmp_path):
    out = rescore_shortlist(
        pd.DataFrame(), FakeBackend(), str(tmp_path), _ORIG_SEQS
    )
    assert out.empty


def test_row_order_is_preserved(tmp_path):
    out = rescore_shortlist(
        _shortlist(), FakeBackend(), str(tmp_path), _ORIG_SEQS
    )
    assert list(out["name"]) == ["bb1", "bb2"]


def test_scores_are_matched_by_sample_name_not_position(tmp_path):
    """A positional join would silently mix up designs."""
    backend = FakeBackend()
    shortlist = _shortlist().iloc[::-1].reset_index(drop=True)
    out = rescore_shortlist(shortlist, backend, str(tmp_path), _ORIG_SEQS)
    # FakeBackend scores entries in manifest order; bb2 is first here
    expected = {
        f"{r['name']}_seq{r['seq_idx']}": r["bz_ipsae"] for _, r in out.iterrows()
    }
    assert expected["bb2_seq0"] == 0.5
    assert expected["bb1_seq0"] == 0.6
