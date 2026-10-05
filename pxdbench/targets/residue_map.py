"""Source-to-working residue provenance, with a deliberately bounded contract.

`pxdesign.utils.residue_numbering.build_residue_maps` already gives a bijective
`res_id <-> auth_res_id` map with uniqueness checks in both directions, and this
module reuses it rather than reimplementing it. It cannot help a pre-built shard
for exactly one reason: **the shard's `auth_res_id` equals its `res_id`**, so the
map it builds is the identity and carries no provenance. Measured on obj2,
`auth_seq_id` is the string form of `res_id` for all 183 residues.

What this adds is the two things that map lacks:

- **real author numbering**, taken from the supplied source structure;
- **residue identity**, without which no identity assertion is possible, so
  "we targeted His535" stays a claim rather than a check.

The supplied source may be a FRAGMENT in biological numbering - obj2's begins at
author residue 512 - so pairing compares the whole selected chain in the source
as supplied and never guesses a full-length protein. No constant-offset
assumption is used, even though obj2 happens to have clean +226 / +511 offsets:
one insertion would break that silently.

V1 is bounded. Every entry in the targeting-contracts section 2 rejection table
fails here, before model initialisation, naming what it found. Chain selection
is supported; residue cropping is not.

Imports only the standard library and the extracted numbering helpers, so the
map contracts are testable without initialising a model.
"""
import gzip
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

from pxdbench.targets.registry import _digest

#: The 20 standard amino acids. A residue outside this set is rejected rather
#: than silently converted - an MSE is not a MET for the purpose of an identity
#: assertion.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

#: Excluded explicitly from protein-residue pairing, never used to repair an
#: incomplete protein chain.
NON_POLYMER = {"HOH", "WAT", "DOD"}


class UnsupportedMapError(ValueError):
    """The supplied structures are outside the v1 contract."""


class AmbiguousChainError(UnsupportedMapError):
    """More than one source chain pairs with a work chain."""


class ResidueNotFound(KeyError):
    """A requested residue is not in the map."""


@dataclass(frozen=True)
class ResidueRow:
    auth_chain: str
    auth_res_id: int
    work_chain: str
    work_res_id: int
    #: 1-based position in the sequence emitted to the predictor. Stored even
    #: though it equals `work_res_id` in this bounded version, because a Boltz
    #: pocket contact is an index into that sequence and the equality is a v1
    #: property rather than a law.
    seq_index: int
    resname: str


def _residue_list(entries, where):
    """Normalise and validate one chain's residues.

    entries: [(chain, res_id, resname, ins_code), ...]
    """
    seen = {}
    ordered = []
    for chain, res_id, resname, ins_code in entries:
        if resname in NON_POLYMER:
            continue
        if str(ins_code or "").strip():
            raise UnsupportedMapError(
                f"{where}: residue {chain}{res_id} has insertion code "
                f"{ins_code!r}. v1 does not support insertion codes; author "
                f"numbering with insertion codes cannot be represented as a "
                f"plain integer without losing identity."
            )
        if resname not in THREE_TO_ONE:
            raise UnsupportedMapError(
                f"{where}: residue {chain}{res_id} is {resname!r}, not one of "
                f"the 20 standard amino acids. v1 rejects it rather than "
                f"silently converting it to a standard residue."
            )
        key = (str(chain), int(res_id))
        if key in seen:
            raise UnsupportedMapError(
                f"{where}: duplicate residue key {key} "
                f"({seen[key]!r} then {resname!r})"
            )
        seen[key] = resname
        ordered.append((str(chain), int(res_id), str(resname)))
    return ordered


def _require_contiguous(entries, work_chain):
    ids = [r for _c, r, _n in entries]
    expected = list(range(1, len(ids) + 1))
    if ids != expected:
        raise UnsupportedMapError(
            f"work chain {work_chain!r} has res_id values "
            f"{ids[:4]}{'...' if len(ids) > 4 else ''} spanning "
            f"{min(ids)}..{max(ids)}; v1 requires contiguous 1..L. Shifted or "
            f"gapped ids mean a crop or a renumbering this version will not "
            f"guess an offset for."
        )


def _pair_chain(source_chains, work_entries, work_chain, explicit):
    """Which source chain this work chain is a renumbered copy of."""
    work_names = [n for _c, _r, n in work_entries]
    if explicit is not None:
        if explicit not in source_chains:
            raise UnsupportedMapError(
                f"chain table maps work chain {work_chain!r} to source chain "
                f"{explicit!r}, which is not in the source "
                f"{sorted(source_chains)}"
            )
        candidates = [explicit]
    else:
        candidates = [
            name
            for name, entries in sorted(source_chains.items())
            if [n for _c, _r, n in entries] == work_names
        ]
    if not candidates:
        raise UnsupportedMapError(
            f"no source chain pairs with work chain {work_chain!r} "
            f"({len(work_entries)} residues). Source chains offered: "
            + ", ".join(
                f"{n} ({len(e)} residues)"
                for n, e in sorted(source_chains.items())
            )
        )
    if len(candidates) > 1:
        raise AmbiguousChainError(
            f"work chain {work_chain!r} pairs with more than one source chain: "
            f"{candidates}. v1 does not choose the first or infer an "
            f"assignment - supply an explicit chain table."
        )
    return candidates[0]


class TargetResidueMap:
    """Rows pairing a shard's working numbering to its source's author numbering."""

    def __init__(self, rows, chain_table):
        self.rows = tuple(rows)
        self.chain_table = dict(chain_table)
        self._by_auth = {(r.auth_chain, r.auth_res_id): r for r in self.rows}
        self._by_work = {(r.work_chain, r.work_res_id): r for r in self.rows}

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_residue_lists(
        cls,
        source,
        shard,
        chain_filter,
        sequences,
        chain_table=None,
    ):
        """Build from duck-typed residue lists.

        source / shard: {chain: [(chain, res_id, resname, ins_code), ...]}
        sequences:      {work_chain: the one-letter sequence emitted to the
                        predictor}. Checked for exact agreement, because a
                        pocket contact is a 1-based index into it.
        """
        clean_source = {
            name: _residue_list(entries, f"source chain {name}")
            for name, entries in source.items()
        }
        explicit = dict(chain_table or {})
        if explicit and len(set(explicit.values())) != len(explicit):
            raise UnsupportedMapError(
                f"chain table assigns one source chain to several work chains: "
                f"{explicit}. A pairing must be a bijection."
            )

        rows, resolved_table = [], {}
        for work_chain in chain_filter:
            if work_chain not in shard:
                raise UnsupportedMapError(
                    f"work chain {work_chain!r} is not in the shard "
                    f"{sorted(shard)}"
                )
            work_entries = _residue_list(
                shard[work_chain], f"work chain {work_chain}"
            )
            _require_contiguous(work_entries, work_chain)

            auth_chain = _pair_chain(
                clean_source, work_entries, work_chain, explicit.get(work_chain)
            )
            source_entries = clean_source[auth_chain]
            if len(source_entries) != len(work_entries):
                raise UnsupportedMapError(
                    f"work chain {work_chain!r} has {len(work_entries)} "
                    f"residues but source chain {auth_chain!r} has "
                    f"{len(source_entries)} residues; the shard is not a "
                    f"complete renumbered copy of that chain"
                )

            emitted = sequences.get(work_chain)
            if emitted is None:
                raise UnsupportedMapError(
                    f"no emitted sequence supplied for work chain "
                    f"{work_chain!r}; the map must agree with what the "
                    f"predictor is given"
                )
            if len(emitted) != len(work_entries):
                raise UnsupportedMapError(
                    f"work chain {work_chain!r}: the emitted sequence is "
                    f"{len(emitted)} residues but the structure has "
                    f"{len(work_entries)}. A pocket contact is an index into "
                    f"that sequence, so they must agree exactly."
                )

            for i, ((_wc, w_res, w_name), (_ac, a_res, a_name)) in enumerate(
                zip(work_entries, source_entries)
            ):
                if w_name != a_name:
                    raise UnsupportedMapError(
                        f"identity mismatch pairing work {work_chain}{w_res} "
                        f"({w_name}) with source {auth_chain}{a_res} "
                        f"({a_name}) at position {i + 1}"
                    )
                if THREE_TO_ONE[w_name] != emitted[i]:
                    raise UnsupportedMapError(
                        f"work {work_chain}{w_res} is {w_name} "
                        f"({THREE_TO_ONE[w_name]}) but the emitted sequence "
                        f"has {emitted[i]!r} at position {i + 1}"
                    )
                rows.append(
                    ResidueRow(
                        auth_chain=auth_chain,
                        auth_res_id=a_res,
                        work_chain=work_chain,
                        work_res_id=w_res,
                        seq_index=i + 1,
                        resname=w_name,
                    )
                )
            resolved_table[work_chain] = auth_chain

        if len(set(resolved_table.values())) != len(resolved_table):
            raise UnsupportedMapError(
                f"two work chains paired with the same source chain: "
                f"{resolved_table}"
            )
        return cls(rows, resolved_table)

    @classmethod
    def from_structures(cls, source_path, shard_path, chain_filter,
                        chain_table=None):
        """Build from a source structure file and a `.pkl.gz` shard."""
        return cls.from_residue_lists(
            source=_read_structure_residues(source_path),
            shard=_read_shard_residues(shard_path),
            chain_filter=list(chain_filter),
            sequences=_read_shard_sequences(shard_path),
            chain_table=chain_table,
        )

    # -- lookups ----------------------------------------------------------- #
    def to_work(self, auth_chain, auth_res_id):
        key = (str(auth_chain), int(auth_res_id))
        if key not in self._by_auth:
            raise ResidueNotFound(
                f"source residue {auth_chain}{auth_res_id} not found in the "
                f"map ({len(self.rows)} residues over chains "
                f"{sorted(self.chain_table.values())})"
            )
        return self._by_auth[key]

    def to_auth(self, work_chain, work_res_id):
        key = (str(work_chain), int(work_res_id))
        if key not in self._by_work:
            raise ResidueNotFound(
                f"work residue {work_chain}{work_res_id} not found in the map "
                f"({len(self.rows)} residues over chains "
                f"{sorted(self.chain_table)})"
            )
        return self._by_work[key]

    def seq_index(self, work_chain, work_res_id):
        return self.to_auth(work_chain, work_res_id).seq_index

    # -- identity and persistence ------------------------------------------ #
    @property
    def map_digest(self):
        return _digest(
            {
                "schema": 2,
                "chain_table": self.chain_table,
                "rows": [asdict(r) for r in self.rows],
            }
        )

    def to_json(self, path):
        payload = {
            "schema": 2,
            "chain_table": self.chain_table,
            "rows": [asdict(r) for r in self.rows],
            "map_digest": self.map_digest,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)
        return str(path)

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != 2:
            raise UnsupportedMapError(
                f"{path}: map schema {payload.get('schema')!r}, expected 2"
            )
        return cls(
            [ResidueRow(**row) for row in payload["rows"]],
            payload["chain_table"],
        )


# --- structure readers ------------------------------------------------------
# Kept at the bottom and imported lazily so the contract above stays testable
# without biotite or Biopython.

def _read_structure_residues(path):
    """{chain: [(chain, res_id, resname, ins_code)]} from a PDB or CIF."""
    from Bio.PDB import MMCIFParser, PDBParser

    parser = (
        MMCIFParser(QUIET=True) if str(path).endswith(".cif")
        else PDBParser(QUIET=True)
    )
    model = parser.get_structure("source", str(path))[0]
    out = {}
    for chain in model:
        entries = [
            (chain.id, residue.id[1], residue.get_resname(),
             str(residue.id[2] or "").strip())
            for residue in chain
            if residue.id[0] == " "
        ]
        if entries:
            out[chain.id] = entries
    return out


def _load_shard(path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _read_shard_residues(path):
    payload = _load_shard(path)
    array = payload["atom_array"]
    out = {}
    seen = set()
    for chain_id, res_id, res_name, ins in zip(
        array.chain_id,
        array.res_id,
        array.res_name,
        getattr(array, "ins_code", [""] * len(array.res_id)),
    ):
        key = (str(chain_id), int(res_id))
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(str(chain_id), []).append(
            (str(chain_id), int(res_id), str(res_name), str(ins or "").strip())
        )
    return out


def _read_shard_sequences(path):
    """{work_chain: emitted one-letter sequence}, from the shard's own record."""
    payload = _load_shard(path)
    array = payload["atom_array"]
    sequences = payload.get("sequences", {})
    out = {}
    for chain_id, entity_id in zip(array.chain_id, array.label_entity_id):
        out.setdefault(str(chain_id), str(entity_id))
    return {
        chain: sequences[entity]
        for chain, entity in out.items()
        if entity in sequences
    }
