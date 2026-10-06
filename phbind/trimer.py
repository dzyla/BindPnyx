"""Trimer-aware (N-copy homo-oligomer) primitives for the binder funnel.

Campaign : Anthropic x Adaptyv 2026, Challenge 2 -- de novo binders to human TNF-alpha.
Target   : soluble homotrimer, 3 identical chains. The receptor-binding site sits in the
           groove BETWEEN ADJACENT PROTOMERS, so every co-folding input must contain the
           full trimer and the binder spans two target chains.

WHY THIS FILE EXISTS
--------------------
funnel/common.py and cmp/score_cmp.py were written for a TWO-chain complex
(target = chain A, binder = chain B).  Three things break on a 3-copy target and all
three break SILENTLY, producing a plausible number instead of an error:

 1. ``hotspot_contacts()`` / ``site_occlusion()`` in common.py do
    ``ids = sorted(ch); tgt, bnd = ch[ids[0]], ch[ids[1]]``.
    On chains A,B,C(target) + D(binder) that makes TARGET CHAIN B the "binder":
    every contact/occlusion number is then target-vs-target.
 2. ``ipsae(pae, n_target, n_binder)`` / ``pae_interface_min(pae, nt)`` take the split
    point positionally.  With the EGFR-era constant (nt=621) on a TNF trimer (nt=471)
    the "interface" block of the PAE matrix is a slice of the target itself.
 3. The target chain count is never asserted, so a YAML that accidentally carries one or
    two copies still scores.

Everything here derives the grouping from the PREDICTED STRUCTURE (file order == token
order == PAE row order) and from SEQUENCE IDENTITY, never from chain letters, and
asserts the expected residue count.  ipSAE is computed with the three target chains as
ONE group and the binder as the other.

Numbering convention used campaign-wide (do not change):
  human local 1-157  == UniProt P01375 77-233   (local i -> P01375 76+i)
  mouse local 1-156  == UniProt P06804 80-235   (local i -> P06804 79+i)

Python >= 3.9, numpy only.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------------------- constants
TNF_HUMAN = ("VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTI"
             "SRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL")
TNF_MOUSE = ("LRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGMDLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVS"
             "RFAISYQEKVNLLSAVKSPCPKDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL")
TARGETS = {"human": TNF_HUMAN, "mouse": TNF_MOUSE}
N_COPIES = 3

# ---------------------------------------------------------------------------------------
# TWO DIFFERENT RESIDUE COUNTS. They are not interchangeable and confusing them is the
# single easiest way to produce a plausible wrong number in this campaign.
#
#  (a) THE CANONICAL TARGET PDB  (organiser file, adopted verbatim) contains only the
#      residues OBSERVED in the crystal: the disordered N-terminal tails are ABSENT.
#      human 152/chain (local 1-5 VRSSS missing), mouse 148/chain (local 1-8 missing).
#      -> 456 and 444.  Check with assert_target_pdb().
#
#  (b) A CO-FOLDED COMPLEX from an oracle (Boltz-2 / AF3 / Protenix) contains the
#      FULL-LENGTH construct sequence, because that is what the oracle is given and it
#      models the tails itself.  human 157/chain, mouse 156/chain -> 471 and 468.
#      Check with group_indices().
# ---------------------------------------------------------------------------------------

#: (b) co-folded complex / full-length construct sequence handed to the oracles
EXPECT_TARGET_RESIDUES = {"human": 3 * 157, "mouse": 3 * 156}        # 471, 468
#: (a) canonical organiser target PDB, crystallographically observed residues only
EXPECT_TARGET_PDB_RESIDUES = {"human": 3 * 152, "mouse": 3 * 148}    # 456, 444
OBSERVED_PER_CHAIN = {"human": 152, "mouse": 148}
#: author residue numbers present in the canonical target PDB (inclusive)
AUTHOR_RANGE = {"human": (6, 157), "mouse": (9, 157)}
#: construct-local indices absent from the canonical target PDB (disordered tails)
MISSING_LOCAL = {"human": list(range(1, 6)), "mouse": list(range(1, 9))}
#: 2TNF is numbered to track the HUMAN sequence, so author number 73 is vacant
MOUSE_VACANT_AUTHOR = 73
UNIPROT_OFFSET = {"human": 76, "mouse": 79}                          # local i -> offset + i
#: organiser chain letters are NOT aligned between species (CA superposition, epitope track)
CHAIN_EQUIVALENT = {"human_to_mouse": {"A": "C", "B": "A", "C": "B"},
                    "mouse_to_human": {"C": "A", "A": "B", "B": "C"}}

assert len(TNF_HUMAN) == 157 and len(TNF_MOUSE) == 156
assert EXPECT_TARGET_RESIDUES == {"human": 471, "mouse": 468}
assert EXPECT_TARGET_PDB_RESIDUES == {"human": 456, "mouse": 444}


def author_to_local(species, author):
    """Author residue number in the canonical target PDB -> construct-local index.

    human: identity. mouse: author <= 72 -> author; author >= 74 -> author - 1
    (author 73 is vacant because 2TNF is numbered against the human sequence)."""
    author = int(author)
    if species == "human":
        return author
    if species != "mouse":
        raise GroupingError("species must be human|mouse, got %r" % species)
    if author == MOUSE_VACANT_AUTHOR:
        raise GroupingError("mouse author residue number 73 is vacant in 2TNF/the organiser "
                            "file; there is no such residue")
    return author if author < MOUSE_VACANT_AUTHOR else author - 1


def local_to_author(species, local):
    """Construct-local index -> author residue number in the canonical target PDB."""
    local = int(local)
    if species == "human":
        return local
    if species != "mouse":
        raise GroupingError("species must be human|mouse, got %r" % species)
    return local if local < MOUSE_VACANT_AUTHOR else local + 1


def to_uniprot(species, local):
    return UNIPROT_OFFSET[species] + int(local)

AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X"}


class GroupingError(RuntimeError):
    """Raised whenever the target/binder grouping cannot be established with certainty.

    Never caught internally: a wrong grouping is worse than no number at all."""


# --------------------------------------------------------------------------- structure parsing
def _chains_from_pdb(path):
    chains, order = {}, []
    for line in open(path):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        comp = line[17:20].strip()
        if comp not in AA3to1:
            continue
        cid = line[21]
        key = (line[22:26].strip(), line[26].strip())
        if cid not in chains:
            chains[cid] = []
            order.append(cid)
        if not chains[cid] or chains[cid][-1][0] != key:
            chains[cid].append((key, comp))
    return [(cid, chains[cid]) for cid in order]


def _chains_from_cif(path):
    """Parse the _atom_site loop keeping FILE ORDER (== Boltz/AF3 token order)."""
    cols, rows, in_loop, header = [], [], False, False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                in_loop, header = True, True
                cols.append(s.split(".", 1)[1])
                continue
            if header and not s.startswith("_atom_site."):
                header = False
            if in_loop:
                if s.startswith("#") or s.startswith("loop_") or s.startswith("_") or not s:
                    if rows:
                        break
                    continue
                rows.append(s.split())
    if not cols or not rows:
        raise GroupingError(f"no _atom_site loop parsed from {path}")
    idx = {c: i for i, c in enumerate(cols)}
    cid_col = idx.get("auth_asym_id", idx.get("label_asym_id"))
    seq_col = idx.get("auth_seq_id", idx.get("label_seq_id"))
    comp_col = idx.get("label_comp_id", idx.get("auth_comp_id"))
    ins_col = idx.get("pdbx_PDB_ins_code")
    if cid_col is None or seq_col is None or comp_col is None:
        raise GroupingError(f"{path}: _atom_site is missing chain/seq/comp columns ({cols})")
    chains, order = {}, []
    for r in rows:
        if len(r) <= max(cid_col, seq_col, comp_col):
            continue
        comp = r[comp_col].strip('"')
        if comp not in AA3to1:
            continue
        cid = r[cid_col].strip('"')
        key = (r[seq_col], r[ins_col] if ins_col is not None and len(r) > ins_col else "?")
        if cid not in chains:
            chains[cid] = []
            order.append(cid)
        if not chains[cid] or chains[cid][-1][0] != key:
            chains[cid].append((key, comp))
    return [(cid, chains[cid]) for cid in order]


def chain_table(path):
    """[{chain_id, seq, n_res, start, stop}] in FILE order (== PAE row order).

    `start`/`stop` are 0-based half-open token indices into the PAE matrix."""
    raw = _chains_from_cif(path) if str(path).endswith((".cif", ".mmcif")) else _chains_from_pdb(path)
    out, cursor = [], 0
    for cid, res in raw:
        seq = "".join(AA3to1.get(c, "X") for _, c in res)
        out.append(dict(chain_id=cid, seq=seq, n_res=len(res),
                        start=cursor, stop=cursor + len(res),
                        resnums=[k[0] for k, _ in res]))
        cursor += len(res)
    return out


# --------------------------------------------------------------------------- grouping
def group_indices(path, species=None, target_seq=None, n_copies=N_COPIES,
                  binder_seq=None, expect_target_residues=None, pae_n=None):
    """Split a co-folded complex into ONE target group (all homo-oligomer copies) and
    ONE binder group, using sequence identity -- NOT chain letters, NOT chain order.

    Returns (target_idx, binder_idx, info).  Raises GroupingError on any ambiguity.

    species : 'human' | 'mouse' -> uses TARGETS[species] and EXPECT_TARGET_RESIDUES.
    pae_n   : if given, asserts the token count equals the PAE matrix dimension.
    """
    if target_seq is None:
        if species not in TARGETS:
            raise GroupingError(f"species must be one of {sorted(TARGETS)}, got {species!r}")
        target_seq = TARGETS[species]
    if expect_target_residues is None and species in EXPECT_TARGET_RESIDUES:
        expect_target_residues = EXPECT_TARGET_RESIDUES[species]

    tab = chain_table(path)
    if not tab:
        raise GroupingError(f"{path}: no protein chains parsed")

    tgt_chains = [c for c in tab if c["seq"] == target_seq]
    rest = [c for c in tab if c["seq"] != target_seq]

    if len(tgt_chains) != n_copies:
        near = [(c["chain_id"], c["n_res"],
                 round(sum(a == b for a, b in zip(c["seq"], target_seq)) / max(len(target_seq), 1), 3))
                for c in tab]
        hint = ""
        if (species in OBSERVED_PER_CHAIN and len(tab) == n_copies
                and all(c["n_res"] == OBSERVED_PER_CHAIN[species] for c in tab)):
            hint = (f" -- all {n_copies} chains have {OBSERVED_PER_CHAIN[species]} residues, which is "
                    f"the CANONICAL ORGANISER TARGET PDB (disordered tails absent). "
                    f"group_indices() is for an ORACLE PREDICTION; validate the target file with "
                    f"assert_target_pdb(path, '{species}') instead.")
        raise GroupingError(
            f"CO-FOLDED COMPLEX CHECK failed for {path}: expected {n_copies} chains whose sequence "
            f"is EXACTLY the {species or 'given'} construct ({len(target_seq)} aa); found "
            f"{len(tgt_chains)}. chains (id, n_res, identity_to_construct) = {near}{hint}")
    if len(rest) != 1:
        raise GroupingError(
            f"{path}: expected exactly 1 non-target chain (the binder); found {len(rest)}: "
            f"{[(c['chain_id'], c['n_res']) for c in rest]}")
    binder = rest[0]
    if binder_seq is not None and binder["seq"] != binder_seq:
        raise GroupingError(f"{path}: binder chain {binder['chain_id']} sequence does not match "
                            f"the expected design sequence ({binder['n_res']} vs {len(binder_seq)} aa)")

    target_idx = np.concatenate([np.arange(c["start"], c["stop"]) for c in tgt_chains])
    binder_idx = np.arange(binder["start"], binder["stop"])

    if expect_target_residues is not None and len(target_idx) != expect_target_residues:
        hint = ""
        if species in EXPECT_TARGET_PDB_RESIDUES and len(target_idx) == EXPECT_TARGET_PDB_RESIDUES[species]:
            hint = (f" -- that is exactly EXPECT_TARGET_PDB_RESIDUES['{species}'] "
                    f"({EXPECT_TARGET_PDB_RESIDUES[species]}), so this looks like the CANONICAL "
                    f"TARGET PDB, not a co-folded complex. Use assert_target_pdb() for the target "
                    f"file; group_indices() is for an oracle PREDICTION, whose target chains carry "
                    f"the FULL-LENGTH construct sequence including the disordered tails.")
        raise GroupingError(
            f"CO-FOLDED COMPLEX CHECK failed for {path}: target group has {len(target_idx)} "
            f"residues, expected {expect_target_residues} "
            f"({N_COPIES} x {len(target_seq)} = the full-length construct the oracles receive)"
            f"{hint}")
    n_tok = sum(c["n_res"] for c in tab)
    if pae_n is not None and n_tok != pae_n:
        raise GroupingError(f"{path}: structure has {n_tok} protein residues but the PAE matrix "
                            f"is {pae_n}x{pae_n}; token/residue mapping is not 1:1 "
                            f"(non-protein entities present?)")
    if set(target_idx) & set(binder_idx):
        raise GroupingError(f"{path}: target and binder index sets overlap")

    info = dict(
        structure=str(path), species=species,
        target_chains=[c["chain_id"] for c in tgt_chains],
        binder_chain=binder["chain_id"], binder_len=binder["n_res"],
        binder_seq=binder["seq"],
        n_target_residues=int(len(target_idx)), n_copies=len(tgt_chains),
        chain_order=[c["chain_id"] for c in tab],
        chain_lengths={c["chain_id"]: c["n_res"] for c in tab},
        target_contiguous=bool(np.all(np.diff(target_idx) == 1)),
        binder_is_last=bool(binder["stop"] == n_tok),
    )
    return target_idx, binder_idx, info


# --------------------------------------------------------------------------- metrics
def _d0(n):
    return 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8


def _ipsae_dir(pae, a, b, cutoff=10.0):
    """Identical math to funnel/common.py::_ipsae_dir -- only the index sets change."""
    sub = pae[np.ix_(a, b)]
    valid = sub < cutoff
    n0 = valid.sum(1)
    ptm = 1.0 / (1.0 + (sub / _d0(n0)[:, None]) ** 2)
    return float(np.where(n0 > 0, (ptm * valid).sum(1) / np.maximum(n0, 1), 0.0).max())


def ipsae_grouped(pae, target_idx, binder_idx, cutoff=10.0):
    """(min, max) directional ipSAE with the WHOLE target (all copies) as one group."""
    pae = np.asarray(pae, float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise GroupingError(f"PAE must be square, got {pae.shape}")
    x = _ipsae_dir(pae, binder_idx, target_idx, cutoff)
    y = _ipsae_dir(pae, target_idx, binder_idx, cutoff)
    return min(x, y), max(x, y)


def ipsae_per_copy(pae, path, species, cutoff=10.0):
    """ipSAE of the binder against EACH target protomer separately.

    Diagnostic only -- a binder in the inter-protomer groove should score on two copies.
    The campaign ranking metric is ipsae_grouped (whole trimer as one group)."""
    tab = chain_table(path)
    tgt = [c for c in tab if c["seq"] == TARGETS[species]]
    bnd = [c for c in tab if c["seq"] != TARGETS[species]][0]
    b = np.arange(bnd["start"], bnd["stop"])
    out = {}
    for c in tgt:
        t = np.arange(c["start"], c["stop"])
        out[c["chain_id"]] = min(_ipsae_dir(pae, b, t, cutoff), _ipsae_dir(pae, t, b, cutoff))
    return out


def pae_interface_min_grouped(pae, target_idx, binder_idx):
    pae = np.asarray(pae, float)
    return float(min(pae[np.ix_(binder_idx, target_idx)].min(),
                     pae[np.ix_(target_idx, binder_idx)].min()))


def interface_contacts_per_chain(path, species, cutoff=5.0):
    """Heavy-atom contact counts between the binder and each target protomer.

    Returns {chain_id: n_contacting_target_residues, ...} plus 'n_chains_engaged'.
    Descriptive only -- Challenge-1 calibration showed contact/area counts are coin
    flips for RANKING designed binders (AUC 0.507-0.549). Use them to check that a
    binder actually straddles two protomers, never to rank."""
    coords, meta = _heavy_atoms(path)
    tseq = TARGETS[species]
    tab = chain_table(path)
    bnd = [c for c in tab if c["seq"] != tseq]
    if len(bnd) != 1:
        raise GroupingError(f"{path}: {len(bnd)} non-target chains")
    bid = bnd[0]["chain_id"]
    bmask = np.array([m[0] == bid for m in meta])
    if bmask.sum() == 0:
        raise GroupingError(f"{path}: binder chain {bid} has no heavy atoms")
    B = coords[bmask]
    out = {}
    for c in tab:
        if c["seq"] != tseq:
            continue
        hits = set()
        sel = [i for i, m in enumerate(meta) if m[0] == c["chain_id"]]
        T = coords[sel]
        tres = [meta[i][1] for i in sel]
        D = np.linalg.norm(T[:, None, :] - B[None, :, :], axis=-1)
        close = (D <= cutoff).any(1)
        for i, ok in enumerate(close):
            if ok:
                hits.add(tres[i])
        out[c["chain_id"]] = len(hits)
    out["n_chains_engaged"] = int(sum(1 for k, v in out.items() if k != "n_chains_engaged" and v > 0))
    return out


def _heavy_atoms(path):
    """-> (Nx3 coords, [(chain_id, resnum_str, atom_name)]) for protein heavy atoms."""
    coords, meta = [], []
    p = str(path)
    if p.endswith((".cif", ".mmcif")):
        cols, rows, header, in_loop = [], [], False, False
        for line in open(p):
            s = line.strip()
            if s.startswith("_atom_site."):
                in_loop, header = True, True
                cols.append(s.split(".", 1)[1])
                continue
            if in_loop and not s.startswith("_atom_site."):
                if s.startswith(("#", "loop_", "_")) or not s:
                    if rows:
                        break
                    continue
                rows.append(s.split())
        idx = {c: i for i, c in enumerate(cols)}
        ci, si = idx.get("auth_asym_id", idx.get("label_asym_id")), idx.get("auth_seq_id", idx.get("label_seq_id"))
        comp = idx.get("label_comp_id")
        el, an = idx.get("type_symbol"), idx.get("label_atom_id", idx.get("auth_atom_id"))
        xi, yi, zi = idx["Cartn_x"], idx["Cartn_y"], idx["Cartn_z"]
        for r in rows:
            if len(r) <= max(xi, yi, zi, ci, si):
                continue
            if r[comp].strip('"') not in AA3to1:
                continue
            if el is not None and r[el].strip('"').upper() == "H":
                continue
            coords.append((float(r[xi]), float(r[yi]), float(r[zi])))
            meta.append((r[ci].strip('"'), r[si], r[an].strip('"') if an is not None else "?"))
    else:
        for line in open(p):
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[17:20].strip() not in AA3to1:
                continue
            if line[76:78].strip().upper() == "H" or line[12:16].strip().startswith("H"):
                continue
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            meta.append((line[21], line[22:26].strip(), line[12:16].strip()))
    return np.asarray(coords, float), meta


def hotspot_contacts_grouped(path, species, hotspots, cutoff=5.0, numbering="author",
                             chains=None):
    """Fraction of hotspot residues contacted by the binder in ANY protomer, plus the
    per-chain breakdown.

    numbering : 'author' (DEFAULT) -- residue numbers as they appear in the canonical
                organiser target PDB and in the epitope track's hotspot lists
                (human 6-157; mouse 9-157 with 73 vacant).
                'local' -- construct-local 1..157 / 1..156.
    chains    : optional list of chain ids to restrict to, e.g. ["B", "C"] for a hotspot
                set defined on one specific pair of protomers. Default: every target copy.

    Replaces common.py::hotspot_contacts, which picked sorted(chains)[1] as the binder
    and therefore measured target-vs-target on a trimer."""
    if numbering not in ("author", "local"):
        raise GroupingError("numbering must be 'author' or 'local', got %r" % numbering)
    if numbering == "local":
        hotspot_local = [int(h) for h in hotspots]
    else:
        hotspot_local = [author_to_local(species, h) for h in hotspots]
    coords, meta = _heavy_atoms(path)
    tseq = TARGETS[species]
    tab = chain_table(path)
    bnd = [c for c in tab if c["seq"] != tseq]
    if len(bnd) != 1:
        raise GroupingError(
            f"{path}: expected 3 chains equal to the {species} construct and exactly 1 other "
            f"(the binder); found {len(bnd)} other chains "
            f"{[(c['chain_id'], c['n_res']) for c in bnd]}. hotspot_contacts_grouped() runs on a "
            f"CO-FOLDED COMPLEX, not on the bare target PDB (whose chains are "
            f"{OBSERVED_PER_CHAIN[species]} residues, not {len(tseq)}, because the disordered "
            f"tails are absent).")
    bid = bnd[0]["chain_id"]
    B = coords[np.array([m[0] == bid for m in meta])]
    tgt_ids = [c["chain_id"] for c in tab if c["seq"] == tseq]
    # local index -> resnum string, per chain, via position in the chain (1-based)
    per_chain, hit_any = {}, set()
    for c in tab:
        if c["seq"] != tseq:
            continue
        if chains is not None and c["chain_id"] not in chains:
            continue
        # map local index -> the residue number actually present in THIS structure.
        # A co-folded prediction is numbered 1..N (local); the canonical target PDB uses
        # author numbering. Try author first, fall back to local, and fail if neither.
        present = set(c["resnums"])
        want = {}
        for i in hotspot_local:
            for cand in (str(local_to_author(species, i)), str(i)):
                if cand in present:
                    want[cand] = i
                    break
            else:
                raise GroupingError(
                    f"{path}: hotspot local {i} (author {local_to_author(species, i)}, "
                    f"UniProt {to_uniprot(species, i)}) is not present in chain "
                    f"{c['chain_id']} (residue numbers {min(present)}..{max(present)}). "
                    f"If this is the canonical target PDB, local 1-{max(MISSING_LOCAL[species])} "
                    f"are disordered and absent; they must not be used as hotspots.")
        sel = [k for k, m in enumerate(meta) if m[0] == c["chain_id"] and m[1] in want]
        hits = set()
        if sel:
            T = coords[sel]
            D = np.linalg.norm(T[:, None, :] - B[None, :, :], axis=-1).min(1)
            for k, d in zip(sel, D):
                if d <= cutoff:
                    hits.add(want[meta[k][1]])
        per_chain[c["chain_id"]] = sorted(hits)
        hit_any |= hits
    n = len(set(hotspot_local))
    return dict(frac=len(hit_any) / n if n else 0.0, n_hit=len(hit_any), n_hotspots=n,
                per_chain=per_chain, target_chains=tgt_ids, binder_chain=bid,
                hotspots_hit=sorted(hit_any))


# --------------------------------------------------------------------------- YAML + fail-fast
def write_boltz_yaml(path, target_seq, target_msa_abs, binder_seq, n_copies=N_COPIES,
                     binder_cyclic=False, msa_placeholder=None):
    """Boltz-2 YAML for an N-copy target + 1 binder.

    * EVERY target copy gets the SAME `msa:` -- the single canonical target a3m.
    * The binder chain gets `msa: empty` (single sequence, no templates). This is the
      campaign convention and it is not negotiable: a different MSA configuration
      between oracles has previously masqueraded as a scientific finding.
    * `target_msa_abs` MUST be absolute. Boltz-2 resolves `msa:` against the JOB CWD,
      not the YAML's directory, so a relative './x.a3m' beside the YAML fails with
      'MSA file not found'. Pass `msa_placeholder='__MSA_HUMAN__'` to emit a token that
      a launcher seds to an absolute path at job start instead.
    """
    if msa_placeholder is None:
        if not os.path.isabs(str(target_msa_abs)):
            raise GroupingError(f"target MSA path must be absolute, got {target_msa_abs!r}")
        if not os.path.exists(target_msa_abs):
            raise GroupingError(f"target MSA does not exist: {target_msa_abs}")
        q = next((l.strip() for l in open(target_msa_abs) if not l.startswith(">") and l.strip()), "")
        if q.replace("-", "").upper() != target_seq:
            raise GroupingError(f"{target_msa_abs}: first (query) row is not the target construct "
                                f"sequence -- wrong MSA for this target")
        msa_val = str(target_msa_abs)
    else:
        msa_val = msa_placeholder
    ids = [chr(ord("A") + i) for i in range(n_copies)]
    bid = chr(ord("A") + n_copies)
    lines = ["version: 1", "sequences:"]
    for i in ids:
        lines += [f"  - protein:", f"      id: {i}", f"      sequence: {target_seq}",
                  f"      msa: {msa_val}"]
    lines += [f"  - protein:", f"      id: {bid}", f"      sequence: {binder_seq}",
              f"      msa: empty"]
    if binder_cyclic:
        lines.append("      cyclic: true")
    txt = "\n".join(lines) + "\n"
    if path is not None:
        open(path, "w").write(txt)
    return txt


def assert_boltz_produced_output(out_dir, min_n=1):
    """Boltz-2 exits 0 after skipping EVERY design when a chain has no `msa:` key
    ('Missing MSA's in input and --use_msa_server flag not set'). Call this immediately
    after the first design and abort the batch if nothing was written."""
    cj = glob.glob(os.path.join(out_dir, "**", "confidence*.json"), recursive=True)
    pz = glob.glob(os.path.join(out_dir, "**", "pae_*.npz"), recursive=True)
    if len(cj) < min_n or len(pz) < min_n:
        raise GroupingError(
            f"Boltz-2 produced {len(cj)} confidence json and {len(pz)} pae npz under {out_dir} "
            f"(need >= {min_n}). Boltz-2 EXITS 0 in this situation -- the usual cause is a YAML "
            f"chain without an `msa:` key, or an `msa:` path that does not resolve against the "
            f"job CWD. Check the boltz log for \"Missing MSA's in input\".")
    return dict(n_confidence=len(cj), n_pae=len(pz))


def assert_canonical_msa(path, species):
    """The single canonical target MSA invariant: first row == construct sequence."""
    if not os.path.isabs(path):
        raise GroupingError(f"MSA path must be absolute: {path}")
    if not os.path.exists(path):
        raise GroupingError(f"MSA not found: {path}")
    q = next((l.strip() for l in open(path) if not l.startswith(">") and l.strip()), "")
    want = TARGETS[species]
    if q.replace("-", "").upper() != want:
        raise GroupingError(f"{path}: query row ({len(q)} chars) is not the {species} TNF "
                            f"construct ({len(want)} aa). Wrong MSA -- refusing to run.")
    n = sum(1 for l in open(path) if l.startswith(">"))
    return dict(path=path, species=species, depth=n, query_len=len(want))


def assert_target_pdb(path, species, allow_het=False):
    """Validate the CANONICAL ORGANISER TARGET PDB (not a prediction).

    This is the file the organisers evaluate against and the file every generator reads.
    It contains ONLY the crystallographically observed residues: the disordered
    N-terminal tails are ABSENT (human local 1-5, mouse local 1-8). Expected counts are
    therefore 456 (human) and 444 (mouse), NOT the 471/468 of a co-folded complex.

    Returns a dict with the chain table, the author<->local map and the heteroatoms found.
    Raises GroupingError with a message that explicitly distinguishes this check from the
    co-folded-complex check in group_indices().
    """
    if species not in TARGETS:
        raise GroupingError(f"species must be human|mouse, got {species!r}")
    want_chain = OBSERVED_PER_CHAIN[species]
    want_total = EXPECT_TARGET_PDB_RESIDUES[species]
    tab = chain_table(path)
    het = {}
    for line in open(path):
        if line.startswith("HETATM"):
            het[line[17:20].strip()] = het.get(line[17:20].strip(), 0) + 1

    if len(tab) != N_COPIES:
        raise GroupingError(
            f"TARGET PDB CHECK failed for {path}: {len(tab)} protein chains "
            f"{[c['chain_id'] for c in tab]}, expected {N_COPIES} (A, B, C). The receptor site is "
            f"in the groove BETWEEN ADJACENT PROTOMERS and does not exist in a monomer.")
    total = sum(c["n_res"] for c in tab)
    if total != want_total:
        hint = ""
        if total == EXPECT_TARGET_RESIDUES[species]:
            hint = (f" -- that is exactly EXPECT_TARGET_RESIDUES['{species}'] "
                    f"({EXPECT_TARGET_RESIDUES[species]}), i.e. the FULL-LENGTH construct with the "
                    f"disordered tails modelled. This is NOT the canonical organiser target file; "
                    f"it is a rebuild or an oracle prediction. Use group_indices() for those.")
        raise GroupingError(
            f"TARGET PDB CHECK failed for {path}: {total} observed residues "
            f"({ {c['chain_id']: c['n_res'] for c in tab} }), expected {want_total} "
            f"({N_COPIES} x {want_chain} = the crystallographically observed residues of the "
            f"canonical organiser file; local {MISSING_LOCAL[species][0]}-"
            f"{MISSING_LOCAL[species][-1]} are disordered and absent){hint}")
    for c in tab:
        if c["n_res"] != want_chain:
            raise GroupingError(
                f"TARGET PDB CHECK failed for {path}: chain {c['chain_id']} has {c['n_res']} "
                f"residues, expected {want_chain}")

    lo, hi = AUTHOR_RANGE[species]
    out_chains = {}
    for c in tab:
        nums = sorted(int(n) for n in c["resnums"])
        if (nums[0], nums[-1]) != (lo, hi):
            raise GroupingError(
                f"TARGET PDB CHECK failed for {path}: chain {c['chain_id']} author numbering spans "
                f"{nums[0]}-{nums[-1]}, expected {lo}-{hi}")
        gaps = sorted(set(range(lo, hi + 1)) - set(nums))
        want_gaps = [MOUSE_VACANT_AUTHOR] if species == "mouse" else []
        if gaps != want_gaps:
            raise GroupingError(
                f"TARGET PDB CHECK failed for {path}: chain {c['chain_id']} author numbers absent "
                f"in range = {gaps}, expected {want_gaps} "
                f"({'2TNF is numbered against the human sequence so 73 is vacant' if species == 'mouse' else 'no gaps'})")
        # sequence must match the construct at every observed position
        mism = []
        for n, aa in zip(nums, c["seq"]):
            i = author_to_local(species, n)
            if TARGETS[species][i - 1] != aa:
                mism.append(dict(author=n, local=i, uniprot=to_uniprot(species, i),
                                 in_pdb=aa, in_construct=TARGETS[species][i - 1]))
        out_chains[c["chain_id"]] = dict(n_res=c["n_res"], author_range=[nums[0], nums[-1]],
                                         sequence_mismatches=mism)
    if het and not allow_het:
        raise GroupingError(
            f"TARGET PDB CHECK failed for {path}: heteroatoms present {het}. Some generators read "
            f"HETATM records as part of the target. Strip them, or for mouse use "
            f"tnf_target_mouse_noHET.pdb. Pass allow_het=True to accept them deliberately.")
    return dict(path=str(path), species=species, n_chains=len(tab),
                chains=[c["chain_id"] for c in tab], residues_per_chain=want_chain,
                total_residues=total, author_range=list(AUTHOR_RANGE[species]),
                missing_local=MISSING_LOCAL[species],
                missing_uniprot=[to_uniprot(species, i) for i in MISSING_LOCAL[species]],
                author_to_local_rule=("author == local" if species == "human" else
                                      "local = author for author<=72; local = author-1 for "
                                      "author>=74; author 73 vacant"),
                heteroatoms=het, per_chain=out_chains)
