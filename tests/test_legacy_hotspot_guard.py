"""Legacy-mode tasks (no epitope block) used to skip hotspot validation entirely, so a hotspot list in the wrong
numbering (or beyond the structure) ran silently. They now fail loudly when the shard cannot contain them."""
import pytest

from pxdbench.targets.preparation import convert_and_resolve


class _FakeAA:
    def __init__(self, n):
        self.chain_id = ["A"] * n
        self.res_id = list(range(1, n + 1))

    def __len__(self):
        return len(self.res_id)


@pytest.fixture
def shard85(monkeypatch):
    import pxdesign.utils.infer as infer
    monkeypatch.setattr(infer, "load_gzip_pickle", lambda path: {"atom_array": _FakeAA(85)})


def _task(tmp_path, hotspots):
    return {"name": "t", "hotspot": {"A": hotspots}, "condition": {"structure_file": str(tmp_path / "x.pkl.gz")}}


def test_in_range_hotspots_pass_unchanged(tmp_path, shard85):
    out = convert_and_resolve(_task(tmp_path, [30, 37, 38]), str(tmp_path))
    assert out["hotspot"] == {"A": [30, 37, 38]} and out["selection_mode"] == "legacy"


def test_hotspots_beyond_the_shard_raise(tmp_path, shard85):
    """MDM2 in PDB numbering (93, 96) on an 85-residue shard: previously ignored without a word."""
    with pytest.raises(KeyError, match="hotspot residues not found"):
        convert_and_resolve(_task(tmp_path, [54, 61, 62, 93, 96]), str(tmp_path))


def test_no_hotspots_is_untouched(tmp_path):
    out = convert_and_resolve({"name": "t", "condition": {"structure_file": "x.pkl.gz"}}, str(tmp_path))
    assert out["selection_mode"] == "legacy" and not (tmp_path / "resolved_hotspots.json").exists()
