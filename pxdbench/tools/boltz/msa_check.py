"""Does a cached MSA actually apply to the sequence handed to the predictor?

Depth cannot see this. Boltz compares its MSA's first sequence against the input
sequence and, on a mismatch it cannot repair, replaces the whole MSA with a
dummy -- printing one line to stdout in the middle of a progress bar
(`boltz/data/feature/featurizerv2.py`, the `creating dummy` branches). Measured
on this project's reference target: a single K->N substitution discarded all
3,109 sequences, and every prediction on that target ran unconditioned.

The comparison here replicates boltz's own, including its tolerance: a mismatch
set that is entirely (input MET, msa UNK), or entirely input UNK, is repaired by
boltz rather than discarded, so it is not an error here either. An exact string
compare is STRICTER than boltz and would reject a usable MSA -- which is why
this module exists rather than a `==`.

Two depths are reported because they differ: boltz's parser deduplicates on the
gapless upper-case sequence, so `grep -c '^>'` overcounts what it loads.
"""

import hashlib
import os

#: Boltz prints this to STDOUT. That is why `runner.py` must keep stdout: it is
#: the only channel on which a discarded MSA is ever reported.
DUMMY_MSA_WARNING = "Warning: MSA does not match input sequence, creating dummy."

A3M_NAME = "non_pairing.a3m"

#: Below this, a cache looks MSA-conditioned and is not. Tunable per caller;
#: the identity check above is not.
DEFAULT_MIN_DEPTH = 100


class MsaMismatch(ValueError):
    """This MSA would be discarded by boltz. Raised before any GPU work."""


def query_residues(a3m_path):
    """The first sequence as boltz parses it, or None if there is none.

    Boltz's a3m parser skips lowercase insertion columns and keeps '-' as a gap
    token, so neither the raw line nor a gap-stripped copy is the right thing
    to compare against an input sequence.
    """
    with open(a3m_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(">"):
                continue
            return "".join(c for c in line if c == "-" or not c.islower())
    return None


def a3m_depths(a3m_path):
    """(header_count, unique_count). Boltz loads the second, not the first."""
    headers = 0
    seen = set()
    with open(a3m_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                headers += 1
                continue
            seen.add(line.replace("-", "").upper())
    return headers, len(seen)


_CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: What boltz's a3m parser can tokenise. `const.prot_letter_to_token` is keyed
#: by upper-case letters and '-'; lowercase marks an insertion column and is
#: dropped before lookup. Anything else raises KeyError inside the parser.
_A3M_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-."
)


def unparseable_a3m_bytes(a3m_path, limit=5):
    """Characters in `a3m_path` that boltz's parser cannot tokenise.

    Returns [(line_number, repr(char), count), ...], empty when the file is
    clean. The query line matching is NOT sufficient: a single stray byte
    anywhere in the file makes boltz raise KeyError per example, skip every
    design, print to stdout and exit 0. One NUL byte on the last line of a
    4,024-sequence alignment cost a 20-backbone campaign on 2026-10-03, with
    the alignment reported as an exact match throughout.
    """
    bad = []
    with open(a3m_path, "rb") as fh:
        for number, raw in enumerate(fh, start=1):
            line = raw.decode("latin-1").rstrip("\r\n")
            if line.startswith(">") or line.startswith("#"):
                continue
            offenders = {c for c in line if c not in _A3M_ALPHABET}
            for char in sorted(offenders):
                bad.append((number, repr(char), line.count(char)))
                if len(bad) >= limit:
                    return bad
    return bad


def _tokens(s):
    """Boltz compares token ids, not letters.

    `const.prot_letter_to_token` maps B, J, O, U, X and Z all to UNK, so they
    are one residue to boltz. '-' is its own gap token, NOT an UNK, and stays.
    """
    return "".join(c if c in _CANONICAL or c == "-" else "X" for c in s)


def compare(sequence, query):
    """(ok, reason), replicating boltz's comparison and its MET/UNK tolerance.

    Compared as token ids (see `_tokens`); mismatches are REPORTED with the
    original letters, which is what diagnosis needs.
    """
    if query is None:
        return False, "the a3m contains no sequence line"
    if len(sequence) != len(query):
        return False, (
            f"length differs: sequence {len(sequence)} vs msa query "
            f"{len(query)} (boltz warning '2')"
        )
    ns, nq = _tokens(sequence), _tokens(query)
    idx = [i for i in range(len(ns)) if ns[i] != nq[i]]
    if not idx:
        return True, "exact match"
    # Boltz repairs these two cases instead of dummying the MSA.
    if all(ns[i] == "M" and nq[i] == "X" for i in idx) or all(
        ns[i] == "X" for i in idx
    ):
        return True, (
            f"{len(idx)} mismatch(es), all repaired by boltz (MET/UNK)"
        )
    shown = ", ".join(
        f"pos {i} sequence={sequence[i]} msa={query[i]}" for i in idx[:8]
    )
    return False, (
        f"{len(idx)} mismatch(es) -> boltz warning '1', the whole MSA is "
        f"replaced with a dummy: {shown}"
    )


def sequence_sha256(sequence):
    """Hash of the sequence actually handed to the predictor."""
    return hashlib.sha256(sequence.encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_a3m(sequence, a3m, label="chain", min_depth=DEFAULT_MIN_DEPTH):
    """Raise unless this a3m will really condition `sequence`; return a record.

    `a3m` is a PATH TO THE FILE, because that is what the YAML writer receives:
    `target_chains_from_orig_seqs` resolves a `target_msa` override, a
    configurable `a3m_name`, and a string-or-dict `msa` down to one path. A
    validator that re-derived the path from a directory would check a different
    file from the one boltz reads.
    """
    if not os.path.isfile(a3m):
        raise MsaMismatch(
            f"{label}: no such MSA file {a3m!r}. The calibrated configuration "
            f"uses cached MSAs - --use_msa_server is never passed - so this "
            f"file must exist."
        )
    junk = unparseable_a3m_bytes(a3m)
    if junk:
        where = ", ".join(f"line {n}: {c} x{k}" for n, c, k in junk)
        raise MsaMismatch(
            f"{label}: {a3m} contains characters boltz's a3m parser cannot "
            f"tokenise ({where}). It raises KeyError on each one, skips every "
            f"design, and still exits 0 - so the run produces no structures "
            f"and reports success. Strip them before folding."
        )
    headers, unique = a3m_depths(a3m)
    ok, reason = compare(sequence, query_residues(a3m))
    if not ok:
        raise MsaMismatch(
            f"{label}: this cached MSA does not apply to the sequence being "
            f"predicted, so boltz would discard all {unique} sequences and "
            f"predict unconditioned. {reason}"
        )
    if unique < min_depth:
        raise MsaMismatch(
            f"{label}: MSA depth {unique} (from {headers} records) is below "
            f"{min_depth}. A shallow cache looks MSA-conditioned and is not, "
            f"and the failure is invisible in the scores it produces. To use "
            f"it anyway, lower the floor deliberately: "
            f"`tools.boltz.min_msa_depth` in the config (or `min_depth` in a "
            f"panel target spec)."
        )
    return {
        "label": label,
        "a3m": a3m,
        "depth_headers": headers,
        "depth_unique": unique,
        "sequence_len": len(sequence),
        "sequence_sha256": sequence_sha256(sequence),
        "a3m_sha256": file_sha256(a3m),
        "match": reason,
    }


def validate_target_chains(target_chains, min_depth=DEFAULT_MIN_DEPTH):
    """Validate the RESOLVED chain/MSA pairs that the YAML writer receives.

    `target_chains` is `target_chains_from_orig_seqs`'s output:
    `[{"id", "seq", "msa"}, ...]` where `msa` is a full a3m path. Validating
    the entity list instead would miss three real cases - a `target_msa`
    override replacing the entity's MSA, a non-default `a3m_name`, and an
    entity whose `msa` is a bare string - and would also skip a chain with
    `use_msa: False` whose MSA still reaches the YAML, because `msa_path_of`
    never consults that flag.

    There is no "no MSA" case to tolerate here: the resolver already raises
    when nothing resolves for a chain.
    """
    if not isinstance(target_chains, (list, tuple)):
        raise MsaMismatch(
            f"target_chains must be a list of resolved chain dicts, got "
            f"{type(target_chains).__name__}"
        )
    records = []
    for idx, chain in enumerate(target_chains):
        chain = chain or {}
        label = f"target chain {chain.get('id', idx)}"
        sequence = chain.get("seq")
        if not sequence:
            raise MsaMismatch(f"{label}: resolved chain has no sequence")
        a3m = chain.get("msa")
        if not a3m:
            raise MsaMismatch(f"{label}: resolved chain has no MSA path")
        records.append(
            validate_a3m(sequence, a3m, label=label, min_depth=min_depth)
        )
    return records


def validate_msa_dir(sequence, msa_dir, label="chain", min_depth=DEFAULT_MIN_DEPTH):
    """Convenience for the command-line checker: a directory, default filename.

    Production validates `validate_target_chains`, which sees the resolved
    path. This exists so `scripts/check_msa_match.sh` can take a directory the
    way it always has.
    """
    a3m = msa_dir if msa_dir.endswith(".a3m") else os.path.join(msa_dir, A3M_NAME)
    return validate_a3m(sequence, a3m, label=label, min_depth=min_depth)
