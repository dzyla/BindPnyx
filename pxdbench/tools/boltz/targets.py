"""Convert PXDesign's `orig_seqs` into Boltz target chain specs.

`orig_seqs` is a LIST of one-key entity dicts, not a chain-keyed mapping
(pxdesign/runner/inference.py:165-171):

    [{"proteinChain": {"sequence": ..., "count": 1, "label_asym_id": ["A0"]}}, ...]

The binder entity is already popped upstream, so every entry is a target.

Each target entity carries its OWN cached MSA directory, which is where the
per-chain MSA comes from. It is NOT configured by chain name: the design's
target chains are A and B while the cached directories are named after the
SOURCE structure's chains (obj2_chainB, obj2_chainD), so pairing by directory
name silently gives chain A the wrong MSA. Verified on a real completed run.
"""
import os

A3M_DEFAULT = "non_pairing.a3m"


def _entity(seq_entry):
    if not isinstance(seq_entry, dict) or len(seq_entry) != 1:
        raise ValueError(
            f"entity must be a one-key dict, got {type(seq_entry).__name__} "
            f"with keys {list(seq_entry) if isinstance(seq_entry, dict) else '-'}"
        )
    return seq_entry[next(iter(seq_entry))]


def chain_id_of(seq_entry):
    """'A0' -> 'A'. inference.py appends a positional index; strip it."""
    entity = _entity(seq_entry)
    asym = entity.get("label_asym_id")
    if not isinstance(asym, list) or not asym:
        raise ValueError(f"entity has no label_asym_id list: {sorted(entity)}")
    return str(asym[0]).rstrip("0123456789") or str(asym[0])


def msa_path_of(seq_entry, a3m_name=A3M_DEFAULT):
    """The cached a3m for this chain, or None when the entity carries no MSA."""
    entity = _entity(seq_entry)
    msa = entity.get("msa")
    if isinstance(msa, str):
        return msa if msa.endswith(".a3m") else os.path.join(msa, a3m_name)
    if isinstance(msa, dict):
        directory = msa.get("precomputed_msa_dir")
        if directory:
            return os.path.join(directory, a3m_name)
    return None


def target_chains_from_orig_seqs(
    orig_seqs, target_msa_override=None, a3m_name=A3M_DEFAULT
):
    """[{"id","seq","msa"}, ...] for the YAML writer, in entity order."""
    if isinstance(orig_seqs, dict):
        raise TypeError(
            "orig_seqs must be a list of entity dicts (see "
            "pxdesign/runner/inference.py:165-171), not a chain-keyed dict"
        )
    if not orig_seqs:
        raise ValueError("orig_seqs is empty; the target chains are unknown")

    override = target_msa_override or {}
    chains = []
    for seq_entry in orig_seqs:
        entity = _entity(seq_entry)
        chain_id = chain_id_of(seq_entry)
        sequence = entity.get("sequence")
        if not sequence:
            raise ValueError(f"chain {chain_id}: entity has no sequence")
        msa = override.get(chain_id) or msa_path_of(seq_entry, a3m_name)
        if not msa:
            raise ValueError(
                f"chain {chain_id}: no MSA on the entity and no override. The "
                f"calibrated configuration requires the cached target MSA; "
                f"running without it invalidates the gate thresholds."
            )
        chains.append({"id": chain_id, "seq": sequence, "msa": msa})
    return chains
